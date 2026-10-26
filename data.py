"""
Шаг 1. Подготовка последовательностей просмотров (KION) со строгим разделением по времени.
Каждый пользователь представляется цепочкой ID фильмов: [item_1, item_2, ..., item_t], как слова в предложении для GPT-2.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_kion_sequences(
    data_path: str = "../kion_recsys_autofe/data/data_original/interactions.csv",
    cutoff_date: str = "2021-08-08",
    max_users: int = 6000,
    top_k_items: int = 1000,
    max_seq_len: int = 20,
    seed: int = 42,
):
    """
    Загружает датасет KION, делит события строго по времени (до cutoff_date = история/обучение, после = тест)
    и превращает историю каждого пользователя в последовательность токенов (item_id -> 1..V).
    """
    # Поиск файла данных: сначала переданный data_path / переменная окружения, затем типовые относительные пути
    candidate_paths = [
        os.environ.get("KION_DATA_PATH", ""),
        data_path,
        "data/interactions.csv",
        "data/data_original/interactions.csv",
        "interactions.csv",
        os.path.join("..", "kion_recsys_autofe", "data", "data_original", "interactions.csv"),
        os.path.join("..", "data", "interactions.csv"),
    ]
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        # Автоматическое скачивание из открытого репозитория KION, если файл нигде не найден
        print("[Данные] Локальный файл interactions.csv не найден. Загрузка датасета KION из открытого источника...")
        import urllib.request
        import zipfile
        os.makedirs("data", exist_ok=True)
        zip_path = os.path.join("data", "data_original.zip")
        target_csv = os.path.join("data", "interactions.csv")
        url = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("interactions.csv"):
                        with zf.open(member) as src_f, open(target_csv, "wb") as dst_f:
                            dst_f.write(src_f.read())
                        break
            if os.path.exists(zip_path):
                os.remove(zip_path)
            resolved_path = target_csv
        except Exception as e:
            raise FileNotFoundError(
                f"Не удалось найти или автоматически скачать interactions.csv ({e}).\n"
                "Пожалуйста, скачайте датасет вручную и поместите interactions.csv в папку data/."
            )

    data_path = resolved_path

    df = pd.read_csv(
        data_path,
        usecols=["user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"],
    )
    df["watched_pct"] = df["watched_pct"].fillna(0.0).clip(0.0, 100.0)
    df["total_dur"] = df["total_dur"].fillna(0.0).clip(0.0, 36000.0)

    # Оставляем топ-1000 самых популярных фильмов как "словарь токенов" для GPT-2 (1..top_k_items, 0 = PAD)
    top_items = (
        df[df["last_watch_dt"] < cutoff_date]["item_id"]
        .value_counts()
        .head(top_k_items)
        .index.values
    )
    df = df[df["item_id"].isin(top_items)].copy()

    # Сортируем события каждого пользователя строго по времени
    df = df.sort_values(["user_id", "last_watch_dt"]).reset_index(drop=True)

    # Считаем сигнал долгосрочной цели (Retention / продолжение сессии):
    # вернулся ли пользователь в сервис ещё хотя бы на 2 просмотра после текущего фильма?
    df["future_views_count"] = df.groupby("user_id").cumcount(ascending=False)
    df["returned_later"] = (df["future_views_count"] >= 2).astype(np.float32)

    # Два варианта вознаграждения (Reward):
    # 1) Краткосрочное (Engagement): доля досмотра фильма (0..1)
    df["reward_short"] = (df["watched_pct"] / 100.0).astype(np.float32)
    # 2) Долгосрочное (Long-term Satisfaction + Retention): досмотр + бонус за возврат в сервис и длительность
    df["reward_long"] = (
        0.5 * (df["watched_pct"] / 100.0)
        + 0.35 * df["returned_later"]
        + 0.15 * np.minimum(df["total_dur"] / 5400.0, 1.0)
    ).astype(np.float32)

    # Словарь токенов: item_id -> token_id (1 .. V), где 0 зарезервирован под паддинг (PAD)
    item2token = {int(it): idx + 1 for idx, it in enumerate(top_items)}
    df["token_id"] = df["item_id"].map(item2token).astype(np.int64)

    # Разделяем строго по времени: события ДО cutoff_date и ПОСЛЕ cutoff_date
    train_df = df[df["last_watch_dt"] < cutoff_date]
    test_df = df[df["last_watch_dt"] >= cutoff_date]

    # Берём пользователей, у которых есть минимум 3 просмотра в истории до cutoff_date
    user_counts = train_df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= 3].index.values

    rng = np.random.default_rng(seed)
    if len(eligible_users) > max_users:
        sampled_users = rng.choice(eligible_users, size=max_users, replace=False)
    else:
        sampled_users = eligible_users

    train_df = train_df[train_df["user_id"].isin(sampled_users)]
    test_df = test_df[test_df["user_id"].isin(sampled_users)]

    # Группируем истории пользователей до cutoff_date
    hist_by_user = {}
    for uid, grp in train_df.groupby("user_id"):
        hist_by_user[int(uid)] = {
            "tokens": grp["token_id"].values[-max_seq_len:],
            "r_short": grp["reward_short"].values[-max_seq_len:],
            "r_long": grp["reward_long"].values[-max_seq_len:],
        }

    # Будущие просмотры пользователей после cutoff_date (для честной оценки Recall@K и NDCG@K)
    test_targets_by_user = {}
    for uid, grp in test_df.groupby("user_id"):
        u_int = int(uid)
        if u_int in hist_by_user:
            test_targets_by_user[u_int] = list(dict.fromkeys(grp["token_id"].values.tolist()))

    # Делим пользователей с историей на train_users (80%) и val_users (20%) для валидации Reward Model
    all_u = sorted(hist_by_user.keys())
    rng.shuffle(all_u)
    split_idx = int(len(all_u) * 0.8)
    train_users = all_u[:split_idx]
    val_users = all_u[split_idx:]

    print(
        f"[Данные] Словарь: {len(item2token)} фильмов | Юзеров с историей: {len(all_u)} "
        f"(Train: {len(train_users)}, Val: {len(val_users)}) | Юзеров с будущими просмотрами в Test: {len(test_targets_by_user)}"
    )
    return hist_by_user, test_targets_by_user, train_users, val_users, len(item2token)
