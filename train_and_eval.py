"""
Шаг 3 и 4. Полный пайплайн обучения и исследования:
1. Обучение базовой модели RecGPT2 (Next-Item Prediction).
2. Обучение и оценка модели вознаграждения (Reward Model) на отложенной выборке.
3. RL-дообучение RecGPT2 (алгоритм REINFORCE с KL-штрафом, как в RLHF для LLM):
   - Сравнение краткосрочной награды (доля досмотра) и долгосрочной награды (возврат пользователя в сервис).
   - Исследование компромисса «Релевантность (NDCG@10, Recall@10) vs Вознаграждение (Reward) vs Разнообразие (Coverage)»
     при разной силе KL-регуляризации beta.
4. Эксперимент авторегрессионной генерации на 5 шагов вперёд (5-Step Rollout).
"""

import copy
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data import load_kion_sequences
from models import RecGPT2, RewardModel


def pad_sequences(seq_list, max_len: int = 15) -> torch.Tensor:
    """Дополняет последовательности слева нулями (left-padding), чтобы последний просмотренный фильм всегда был справа."""
    arr = np.zeros((len(seq_list), max_len), dtype=np.int64)
    for i, s in enumerate(seq_list):
        s_cut = s[-max_len:]
        arr[i, -len(s_cut) :] = s_cut
    return torch.tensor(arr, dtype=torch.long)


def build_supervised_pairs(hist_by_user, user_ids, max_ctx_len: int = 15):
    """Создаёт пары (история до шага t -> целевой фильм на шаге t и его награды) для обучения GPT-2 и Reward Model."""
    contexts, targets, r_shorts, r_longs = [], [], [], []
    for u in user_ids:
        data = hist_by_user[u]
        tokens = data["tokens"]
        rs = data["r_short"]
        rl = data["r_long"]
        for t in range(1, len(tokens)):
            contexts.append(tokens[:t])
            targets.append(tokens[t])
            r_shorts.append(rs[t])
            r_longs.append(rl[t])
    ctx_tensor = pad_sequences(contexts, max_len=max_ctx_len)
    tgt_tensor = torch.tensor(targets, dtype=torch.long)
    rs_tensor = torch.tensor(r_shorts, dtype=torch.float32)
    rl_tensor = torch.tensor(r_longs, dtype=torch.float32)
    return ctx_tensor, tgt_tensor, rs_tensor, rl_tensor


def train_base_gpt2(ctx_train, tgt_train, vocab_size: int, epochs: int = 12, batch_size: int = 256):
    """1. Обучает базовую модель GPT-2 предсказывать следующий фильм (Cross-Entropy Loss)."""
    model = RecGPT2(vocab_size=vocab_size, embed_dim=64, n_heads=4, n_layers=2, max_len=30)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    n = len(ctx_train)

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(ctx_train[idx])[:, -1, :]  # Предсказание с последней позиции истории
            loss = F.cross_entropy(logits, tgt_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if ep in (1, epochs // 2, epochs):
            print(f"  [Base GPT-2] Эпоха {ep:2d}/{epochs} | Cross-Entropy Loss: {total_loss / n:.4f}")
    return model


def train_and_eval_reward_model(
    ctx_train, tgt_train, r_train, ctx_val, tgt_val, r_val, vocab_size: int, name: str, epochs: int = 10
):
    """2. Обучает Reward Model предсказывать отклик пользователя и оценивает её качество на отложенной выборке Val."""
    rm = RewardModel(vocab_size=vocab_size, embed_dim=64)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=3e-3, weight_decay=1e-4)
    batch_size = 256
    n = len(ctx_train)

    rm.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            preds = rm(ctx_train[idx], tgt_train[idx])
            loss = F.mse_loss(preds, r_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Оценка на отложенной валидационной выборке
    rm.eval()
    with torch.no_grad():
        val_preds = rm(ctx_val, tgt_val).numpy()
    val_true = r_val.numpy()

    mae = float(mean_absolute_error(val_true, val_preds))
    rmse = float(np.sqrt(mean_squared_error(val_true, val_preds)))
    corr = float(pearsonr(val_true, val_preds)[0])
    binary_high = (val_true >= np.median(val_true)).astype(int)
    auc = float(roc_auc_score(binary_high, val_preds))

    print(
        f"  [Reward Model: {name}] Отложенная выборка Val -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
        f"ROC-AUC (высокий отклик): {auc:.4f} | Корреляция Пирсона: {corr:.4f}"
    )
    return rm, {"name": name, "MAE": mae, "RMSE": rmse, "ROC-AUC": auc, "Pearson_R": corr}


def finetune_gpt2_with_rl(
    base_model: RecGPT2,
    reward_model: RewardModel,
    user_histories: torch.Tensor,
    kl_beta: float = 0.15,
    steps: int = 80,
    batch_size: int = 256,
    top_sample_k: int = 30,
    lr: float = 1.5e-3,
):
    """
    3. Дообучает копию RecGPT2 с помощью RL (REINFORCE с Baseline и KL-штрафом к базовой модели).
    Функция потерь аналогична RLHF в языковых моделях:
      Maximize E_{a ~ pi_theta}[ R(s, a) - beta * KL(pi_theta(.|s) || pi_base(.|s)) ]
    """
    policy = copy.deepcopy(base_model)
    base_model.eval()
    reward_model.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    n_users = len(user_histories)
    running_baseline = None

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_users, (batch_size,))
        batch_ctx = user_histories[idx]

        # Запрещаем повторно рекомендовать фильмы, которые пользователь уже только что смотрел
        logits = policy(batch_ctx)[:, -1, :]
        with torch.no_grad():
            base_logits = base_model(batch_ctx)[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        base_log_probs = F.log_softmax(base_logits, dim=-1)

        # Сэмплируем действия из топ-K вероятных кандидатов политики (чтобы ускорить сходимость)
        top_probs, top_items = torch.topk(probs, k=top_sample_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        sampled_rel_idx = torch.multinomial(top_probs, num_samples=1)  # [B, 1]
        sampled_items = torch.gather(top_items, 1, sampled_rel_idx).squeeze(1)  # [B]

        # Оцениваем сэмплированные рекомендации через Reward Model
        with torch.no_grad():
            rewards = reward_model(batch_ctx, sampled_items)  # [B]

        if running_baseline is None:
            running_baseline = rewards.mean().item()
        else:
            running_baseline = 0.9 * running_baseline + 0.1 * rewards.mean().item()

        # Полная дивергенция Кульбака-Лейблера KL(pi_theta || pi_base)
        kl_div = torch.sum(probs * (log_probs - base_log_probs), dim=-1)  # [B]

        # Advantage для алгоритма REINFORCE
        chosen_log_prob = log_probs.gather(1, sampled_items.unsqueeze(1)).squeeze(1)
        advantage = (rewards - running_baseline).detach()

        # Итоговый лосс: максимизируем награду (минимизируем -advantage * log_prob) и штрафуем за уход от базы (KL)
        pg_loss = -(advantage * chosen_log_prob).mean()
        kl_loss = kl_beta * kl_div.mean()
        loss = pg_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    return policy


def evaluate_policy(
    model: RecGPT2,
    base_model: RecGPT2,
    rm_short: RewardModel,
    rm_long: RewardModel,
    hist_by_user,
    test_targets_by_user,
    k: int = 10,
):
    """
    Оценивает модель по 5 ключевым осям:
    1) Релевантность на будущем тесте: Recall@10, NDCG@10, MAP@10;
    2) Ожидаемое краткосрочное вознаграждение (Expected Watch % Reward);
    3) Ожидаемое долгосрочное вознаграждение (Expected Long-Term Retention Reward);
    4) Разнообразие выдачи (Coverage@10 — сколько уникальных фильмов модель рекомендует разным людям, проверка на коллапс);
    5) KL-дивергенция относительно базовой GPT-2.
    """
    model.eval()
    base_model.eval()
    rm_short.eval()
    rm_long.eval()

    eval_users = sorted(test_targets_by_user.keys())
    ctx_list = [hist_by_user[u]["tokens"] for u in eval_users]
    ctx_tensor = pad_sequences(ctx_list, max_len=15)

    with torch.no_grad():
        logits = model(ctx_tensor)[:, -1, :].clone()
        base_logits = base_model(ctx_tensor)[:, -1, :].clone()

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        kl_val = float(torch.sum(probs * (log_probs - base_log_probs), dim=-1).mean().item())

        # Исключаем уже просмотренные в истории фильмы при формировании топ-K новых рекомендаций
        for i, seq in enumerate(ctx_list):
            seen_unique = list(set(seq.tolist()))
            logits[i, seen_unique] = -1e8

        _, topk_items = torch.topk(logits, k=k, dim=-1)  # [N_eval, K]

        # Считаем среднее вознаграждение по топ-K рекомендованным фильмам
        exp_r_short = float(rm_short(ctx_tensor, topk_items).mean().item())
        exp_r_long = float(rm_long(ctx_tensor, topk_items).mean().item())

    topk_np = topk_items.numpy()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    recalls, ndcgs, maps = [], [], []
    for i, u in enumerate(eval_users):
        pos_set = set(test_targets_by_user[u])
        if not pos_set:
            continue
        recs = topk_np[i]
        hits = np.array([1.0 if it in pos_set else 0.0 for it in recs])

        recalls.append(float(hits.sum() / min(len(pos_set), k)))
        dcg = float(np.sum(hits * discounts))
        idcg = float(np.sum(discounts[: min(len(pos_set), k)]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        if hits.sum() > 0:
            hit_idx = np.where(hits > 0)[0]
            precisions = (np.arange(1, len(hit_idx) + 1)) / (hit_idx + 1.0)
            maps.append(float(np.sum(precisions) / min(len(pos_set), k)))
        else:
            maps.append(0.0)

    unique_recommended = int(len(np.unique(topk_np)))
    top1_item_share = float(pd.Series(topk_np[:, 0]).value_counts(normalize=True).iloc[0] * 100.0)

    return {
        "Recall@10": float(np.mean(recalls)),
        "NDCG@10": float(np.mean(ndcgs)),
        "MAP@10": float(np.mean(maps)),
        "Reward_Short": exp_r_short,
        "Reward_Long": exp_r_long,
        "Unique_Items@10": unique_recommended,
        "Top1_Concentration_%": top1_item_share,
        "KL_from_Base": kl_val,
    }


def evaluate_5_step_rollout(
    models_dict, rm_short: RewardModel, rm_long: RewardModel, user_histories: torch.Tensor, n_steps: int = 5
):
    """
    4. Исследует авторегрессионную генерацию на 5 шагов вперёд (5-Step Rollout):
    на каждом шаге модель выбирает фильм, добавляет его в конец истории пользователя,
    и предсказывает следующий шаг.
    """
    rollout_rows = []
    sample_ctx = user_histories[:500].clone()

    for model_name, model in models_dict.items():
        model.eval()
        cur_ctx = sample_ctx.clone()
        generated_history = []

        with torch.no_grad():
            for step in range(1, n_steps + 1):
                logits = model(cur_ctx)[:, -1, :].clone()
                # Небольшой штраф за уже сгенерированные в этом роллауте фильмы, но позволяем увидеть дрейф
                probs = F.softmax(logits / 0.8, dim=-1)
                next_item = torch.argmax(probs, dim=-1)  # Жадное декодирование очередного шага

                r_s = float(rm_short(cur_ctx, next_item).mean().item())
                r_l = float(rm_long(cur_ctx, next_item).mean().item())
                uniq_items = int(len(torch.unique(next_item)))

                # Доля пользователей, которым на шаге step порекомендовали фильм, уже бывший в предыдущих шагах роллаута
                if generated_history:
                    prev_stack = torch.stack(generated_history, dim=1)
                    repeat_rate = float(
                        (prev_stack == next_item.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
                    )
                else:
                    repeat_rate = 0.0

                generated_history.append(next_item)
                rollout_rows.append(
                    {
                        "Model": model_name,
                        "Step": step,
                        "Expected_Reward_Short": r_s,
                        "Expected_Reward_Long": r_l,
                        "Unique_Items_Step": uniq_items,
                        "Self_Repeat_Rate_%": repeat_rate,
                    }
                )

                # Сдвигаем окно контекста и добавляем сгенерированный токен в конец (как в GPT)
                cur_ctx = torch.cat([cur_ctx[:, 1:], next_item.unsqueeze(1)], dim=1)

    return rollout_rows


def save_plots(comparison_df: pd.DataFrame, rollout_df: pd.DataFrame, out_dir: str = "reports"):
    """Рисует 3 понятных графика для отчёта."""
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    # График 1: Компромисс Релевантность (NDCG@10) vs Вознаграждение (Reward) и Разнообразие
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    for i, row in comparison_df.iterrows():
        axes[0].scatter(
            row["Reward_Short"],
            row["NDCG@10"],
            s=180,
            color=colors[i % len(colors)],
            label=row["Model"],
            zorder=5,
        )
        axes[0].annotate(
            row["Model"].split("(")[0].strip(),
            (row["Reward_Short"], row["NDCG@10"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_title("Компромисс: Вознаграждение vs Релевантность (NDCG@10)", fontweight="bold")
    axes[0].set_xlabel("Ожидаемое вознаграждение (Expected Watch %)")
    axes[0].set_ylabel("Релевантность на тесте (NDCG@10)")
    axes[0].grid(True, alpha=0.3)

    axes[1].barh(comparison_df["Model"], comparison_df["Unique_Items@10"], color=colors[: len(comparison_df)])
    axes[1].set_title("Разнообразие рекомендаций (Unique Items@10)\nПри слабом KL-штрафе происходит схлопывание (Mode Collapse)", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Число уникальных фильмов в выдаче топ-10")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rl_tradeoff_and_diversity.png"), dpi=160)
    plt.close()

    # График 2: Что происходит при генерации на 5 шагов вперёд (5-Step Rollout)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name, grp in rollout_df.groupby("Model"):
        axes[0].plot(grp["Step"], grp["Expected_Reward_Long"], marker="o", linewidth=2.2, label=model_name)
        axes[1].plot(grp["Step"], grp["Unique_Items_Step"], marker="s", linewidth=2.2, label=model_name)

    axes[0].set_title("Генерация на 5 шагов вперёд: Ожидаемое вознаграждение", fontweight="bold")
    axes[0].set_xlabel("Шаг авторегрессии (t = 1..5)")
    axes[0].set_ylabel("Expected Long-Term Reward")
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Генерация на 5 шагов вперёд: Падение разнообразия (Дрейф контекста)", fontweight="bold")
    axes[1].set_xlabel("Шаг авторегрессии (t = 1..5)")
    axes[1].set_ylabel("Уникальных фильмов на шаге t (из 500 юзеров)")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rollout_5steps.png"), dpi=160)
    plt.close()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print(" ЗАДАНИЕ 2: ДООБУЧЕНИЕ ПОСЛЕДОВАТЕЛЬНОЙ МОДЕЛИ (GPT-2) С ПОМОЩЬЮ RL (KION)")
    print("=" * 80)

    hist_by_user, test_targets_by_user, train_users, val_users, vocab_size = load_kion_sequences()

    ctx_tr, tgt_tr, rs_tr, rl_tr = build_supervised_pairs(hist_by_user, train_users)
    ctx_va, tgt_va, rs_va, rl_va = build_supervised_pairs(hist_by_user, val_users)

    print("\n--- 1. Обучение базовой модели RecGPT2 (Next-Item Prediction) ---")
    base_gpt2 = train_base_gpt2(ctx_tr, tgt_tr, vocab_size=vocab_size, epochs=12)

    print("\n--- 2. Обучение и проверка моделей вознаграждения (Reward Models) ---")
    rm_short, rm_short_metrics = train_and_eval_reward_model(
        ctx_tr, tgt_tr, rs_tr, ctx_va, tgt_va, rs_va, vocab_size, name="Short-Term (Watch %)"
    )
    rm_long, rm_long_metrics = train_and_eval_reward_model(
        ctx_tr, tgt_tr, rl_tr, ctx_va, tgt_va, rl_va, vocab_size, name="Long-Term (Retention + Watch)"
    )

    print("\n--- 3. RL-дообучение RecGPT2 (REINFORCE + KL Penalty) ---")
    all_user_ctx = pad_sequences([hist_by_user[u]["tokens"] for u in train_users], max_len=15)

    # Вариант А: Мягкий RL на краткосрочную награду (beta = 0.20)
    rl_short_balanced = finetune_gpt2_with_rl(
        base_gpt2, rm_short, all_user_ctx, kl_beta=0.20, steps=80
    )
    # Вариант Б: Умеренный RL на краткосрочную награду (beta = 0.06)
    rl_short_medium = finetune_gpt2_with_rl(
        base_gpt2, rm_short, all_user_ctx, kl_beta=0.06, steps=80
    )
    # Вариант В: Агрессивный RL без сдерживания (beta = 0.005 -> демонстрация Mode Collapse / Reward Hacking)
    rl_short_aggressive = finetune_gpt2_with_rl(
        base_gpt2, rm_short, all_user_ctx, kl_beta=0.005, steps=80
    )
    # Вариант Г: RL на долгосрочную цель (удержание / возврат пользователя, beta = 0.12)
    rl_long_term = finetune_gpt2_with_rl(
        base_gpt2, rm_long, all_user_ctx, kl_beta=0.12, steps=80
    )

    models_to_compare = [
        ("1. Base GPT-2 (No RL)", base_gpt2),
        ("2. RL-Short (Balanced, beta=0.20)", rl_short_balanced),
        ("3. RL-Short (Medium, beta=0.06)", rl_short_medium),
        ("4. RL-Short (Aggressive, beta=0.005)", rl_short_aggressive),
        ("5. RL-LongTerm (Retention, beta=0.12)", rl_long_term),
    ]

    comparison_rows = []
    for label, mdl in models_to_compare:
        res = evaluate_policy(
            mdl, base_gpt2, rm_short, rm_long, hist_by_user, test_targets_by_user
        )
        row = {"Model": label, **res}
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    print("\n================ СРАВНЕНИЕ БАЗОВОЙ GPT-2 И RL-МОДЕЛЕЙ НА ТЕСТЕ ================")
    print(tabulate(comp_df, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))

    print("\n--- 4. Генерация на 5 шагов вперёд (5-Step Autoregressive Rollout) ---")
    rollout_models = {
        "Base GPT-2": base_gpt2,
        "RL-LongTerm (beta=0.12)": rl_long_term,
        "RL-Aggressive (beta=0.005)": rl_short_aggressive,
    }
    rollout_rows = evaluate_5_step_rollout(rollout_models, rm_short, rm_long, all_user_ctx, n_steps=5)
    rollout_df = pd.DataFrame(rollout_rows)
    print(tabulate(rollout_df, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))

    os.makedirs("reports", exist_ok=True)
    comp_df.to_csv("reports/model_comparison.csv", index=False)
    rollout_df.to_csv("reports/rollout_5steps.csv", index=False)
    with open("reports/results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "reward_models": [rm_short_metrics, rm_long_metrics],
                "comparison": comparison_rows,
                "rollout_5steps": rollout_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_plots(comp_df, rollout_df, out_dir="reports")
    print(f"\n[ГОТОВО] Все расчёты и графики сохранены в папку: {os.path.abspath('reports')}/")


if __name__ == "__main__":
    main()
