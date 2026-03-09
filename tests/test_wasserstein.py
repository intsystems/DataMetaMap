import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import traceback

from data_meta_map.BaseEmbedder import BaseEmbedder
from data_meta_map.WassersteinEmbedder import WassersteinEmbedder


# ============================================================================
# Вспомогательные датасеты для тестов
# ============================================================================

class MockDataset(Dataset):
    """Простой датасет для тестов."""
    def __init__(self, num_samples=100, feature_dim=10, num_classes=5, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MockVectorizedDataset(Dataset):
    """Датасет с уже векторизованными данными (как текстовые эмбеддинги)."""
    def __init__(self, num_samples=100, feature_dim=768, num_classes=3, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ============================================================================
# Тесты BaseEmbedder
# ============================================================================

class ConcreteEmbedder(BaseEmbedder):
    """Конкретная реализация для тестирования абстрактного класса."""
    def preprocess_dataset(self, data, dataset_id=None):
        if isinstance(data, Dataset):
            X = torch.randn(100, 10)
            Y = torch.randint(0, 5, (100,))
            return X.to(self.device), Y.to(self.device)
        return data

    def compute_pairwise_distances(self, datasets, symmetric=True):
        n = sum(len(torch.unique(self.preprocess_dataset(d)[1])) for d in datasets)
        return torch.zeros((n, n), device=self.device)

    def embed_distance_matrix(self, distance_matrix, emb_dim=None):
        emb_dim = emb_dim or self.emb_dim
        n = distance_matrix.shape[0]
        return torch.randn(n, emb_dim, device=self.device)

    def augment_features(self, data, label_embeddings, dataset_idx, class_offsets):
        X, Y = self.preprocess_dataset(data)
        label_emb = label_embeddings[class_offsets[dataset_idx]:class_offsets[dataset_idx+1]]
        return torch.cat([X, label_emb[Y]], dim=1)


def test_base_embedder_init():
    """Тест инициализации BaseEmbedder."""
    print("\n[TEST] BaseEmbedder initialization")
    try:
        embedder = ConcreteEmbedder(emb_dim=2, device='cpu')
        assert embedder.emb_dim == 2
        assert embedder.device == torch.device('cpu')
        assert embedder.max_samples is None
        assert embedder.batch_size == 64
        print("  ✓ BaseEmbedder инициализируется корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_base_embedder_device_property():
    """Тест свойства device."""
    print("\n[TEST] BaseEmbedder device property")
    try:
        embedder = ConcreteEmbedder(emb_dim=2)
        embedder.device = 'cpu'
        assert isinstance(embedder.device, torch.device)
        print("  ✓ Device property работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_base_embedder_get_class_statistics():
    """Тест вычисления статистик классов."""
    print("\n[TEST] BaseEmbedder get_class_statistics")
    try:
        embedder = ConcreteEmbedder(emb_dim=2)
        X = torch.randn(100, 10)
        Y = torch.randint(0, 5, (100,))

        means, covs = embedder.get_class_statistics(X, Y)

        assert means.shape == (5, 10)
        assert covs.shape == (5, 10, 10)
        print("  ✓ Статистики классов вычисляются корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Тесты WassersteinEmbedder
# ============================================================================

def test_wasserstein_embedder_init():
    """Тест инициализации WassersteinEmbedder."""
    print("\n[TEST] WassersteinEmbedder initialization")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        assert embedder.emb_dim == 2
        assert embedder.device == torch.device('cpu')
        assert embedder.gaussian_assumption is True
        assert embedder.diagonal_cov is False
        print("  ✓ WassersteinEmbedder инициализируется корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_embedder_init_custom_params():
    """Тест инициализации с кастомными параметрами."""
    print("\n[TEST] WassersteinEmbedder initialization with custom params")
    try:
        embedder = WassersteinEmbedder(
            emb_dim=5,
            device='cpu',
            max_samples=50,
            batch_size=32,
            gaussian_assumption=False,
            diagonal_cov=True,
            sqrt_niters=10
        )
        assert embedder.emb_dim == 5
        assert embedder.max_samples == 50
        assert embedder.batch_size == 32
        assert embedder.gaussian_assumption is False
        assert embedder.diagonal_cov is True
        assert embedder.sqrt_niters == 10
        print("  ✓ Кастомные параметры устанавливаются корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_preprocess_dataset():
    """Тест препроцессинга датасета."""
    print("\n[TEST] WassersteinEmbedder preprocess_dataset")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        dataset = MockDataset(num_samples=100, feature_dim=10, num_classes=5)

        X, Y = embedder.preprocess_dataset(dataset, dataset_id=0)

        assert X.shape == (100, 10)
        assert Y.shape == (100,)
        assert X.dtype == torch.float32
        assert Y.dtype == torch.long
        assert torch.all(Y >= 0) and torch.all(Y < 5)
        print("  ✓ Препроцессинг датасета работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_preprocess_dataloader():
    """Тест препроцессинга через DataLoader."""
    print("\n[TEST] WassersteinEmbedder preprocess_dataset with DataLoader")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        dataset = MockDataset(num_samples=50, feature_dim=20, num_classes=3)
        loader = DataLoader(dataset, batch_size=16)

        X, Y = embedder.preprocess_dataset(loader, dataset_id=1)

        assert X.shape == (50, 20)
        assert Y.shape == (50,)
        print("  ✓ Препроцессинг через DataLoader работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_preprocess_max_samples():
    """Тест субсэмплирования при max_samples."""
    print("\n[TEST] WassersteinEmbedder preprocess with max_samples")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, max_samples=30)
        dataset = MockDataset(num_samples=100, feature_dim=10, num_classes=5)

        X, Y = embedder.preprocess_dataset(dataset, dataset_id=0)

        assert X.shape[0] == 30
        assert Y.shape[0] == 30
        print("  ✓ Субсэмплирование работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_preprocess_caching():
    """Тест кэширования результатов."""
    print("\n[TEST] WassersteinEmbedder preprocess caching")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        dataset = MockDataset(num_samples=50, feature_dim=10, num_classes=3)

        X1, Y1 = embedder.preprocess_dataset(dataset, dataset_id=0)
        X2, Y2 = embedder.preprocess_dataset(dataset, dataset_id=0)

        assert torch.equal(X1, X2)
        assert torch.equal(Y1, Y2)
        print("  ✓ Кэширование работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_preprocess_vectorized():
    """Тест препроцессинга уже векторизованных данных."""
    print("\n[TEST] WassersteinEmbedder preprocess vectorized data")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        dataset = MockVectorizedDataset(num_samples=100, feature_dim=768, num_classes=5)

        X, Y = embedder.preprocess_dataset(dataset, dataset_id=2)

        assert X.shape == (100, 768)
        assert Y.shape == (100,)
        print("  ✓ Препроцессинг векторизованных данных работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_compute_gaussian_stats():
    """Тест вычисления гауссовских статистик."""
    print("\n[TEST] WassersteinEmbedder _compute_gaussian_stats")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        X = torch.randn(100, 10)
        Y = torch.tensor([0]*50 + [1]*50)

        means, covs, offsets = embedder._compute_gaussian_stats(X, Y)

        assert means.shape == (2, 10)
        assert covs.shape == (2, 10, 10)
        assert offsets == [0, 1]
        print("  ✓ Гауссовские статистики вычисляются корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_compute_gaussian_stats_diagonal():
    """Тест вычисления гауссовских статистик с диагональной ковариацией."""
    print("\n[TEST] WassersteinEmbedder _compute_gaussian_stats diagonal")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, diagonal_cov=True)
        X = torch.randn(100, 10)
        Y = torch.tensor([0]*30 + [1]*30 + [2]*40)

        means, covs, offsets = embedder._compute_gaussian_stats(X, Y)

        assert means.shape == (3, 10)
        assert covs.shape == (3, 10)  # Диагональная матрица
        assert offsets == [0, 1, 2]
        print("  ✓ Диагональная ковариация вычисляется корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_bures_distance_identical():
    """Тест расстояния Бюра для идентичных распределений."""
    print("\n[TEST] WassersteinEmbedder _bures_wasserstein_distance identical")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        mean1 = torch.tensor([0.0, 0.0])
        cov1 = torch.eye(2)
        mean2 = torch.tensor([0.0, 0.0])
        cov2 = torch.eye(2)

        distance = embedder._bures_wasserstein_distance(mean1, cov1, mean2, cov2)

        assert torch.allclose(distance, torch.tensor(0.0), atol=1e-2)
        print("  ✓ Расстояние Бюра для идентичных распределений = 0")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_bures_distance_different_means():
    """Тест расстояния Бюра для распределений с разными средними."""
    print("\n[TEST] WassersteinEmbedder _bures_wasserstein_distance different means")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        mean1 = torch.tensor([0.0, 0.0])
        cov1 = torch.eye(2)
        mean2 = torch.tensor([1.0, 0.0])
        cov2 = torch.eye(2)

        distance = embedder._bures_wasserstein_distance(mean1, cov1, mean2, cov2)

        assert distance > 0.0
        assert torch.allclose(distance, torch.tensor(1.0), atol=1e-5)
        print("  ✓ Расстояние Бюра для разных средних вычисляется корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_pairwise_distances_single():
    """Тест матрицы расстояний для одного датасета."""
    print("\n[TEST] WassersteinEmbedder compute_pairwise_distances single dataset")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, max_samples=30)
        dataset = MockDataset(num_samples=50, feature_dim=10, num_classes=3)

        D = embedder.compute_pairwise_distances([dataset], symmetric=True)

        assert D.shape == (3, 3)
        assert torch.allclose(D.diag(), torch.zeros(3), atol=1e-2)
        assert torch.all(D >= 0)
        assert torch.allclose(D, D.T, atol=1e-2)
        print("  ✓ Матрица расстояний для одного датасета корректна")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_pairwise_distances_multiple():
    """Тест матрицы расстояний для нескольких датасетов."""
    print("\n[TEST] WassersteinEmbedder compute_pairwise_distances multiple datasets")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, max_samples=20)
        ds1 = MockDataset(num_samples=30, feature_dim=10, num_classes=2, seed=42)
        ds2 = MockDataset(num_samples=30, feature_dim=10, num_classes=3, seed=43)

        D = embedder.compute_pairwise_distances([ds1, ds2], symmetric=True)

        assert D.shape == (5, 5)  # 2 + 3 = 5 классов
        assert torch.all(D >= 0)
        print("  ✓ Матрица расстояний для нескольких датасетов корректна")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_embed_distance_matrix():
    """Тест MDS-эмбеддинга матрицы расстояний."""
    print("\n[TEST] WassersteinEmbedder embed_distance_matrix")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        # Создаём простую матрицу расстояний
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        D = torch.cdist(points, points, p=2)

        embeddings = embedder.embed_distance_matrix(D, emb_dim=2)

        assert embeddings.shape == (4, 2)
        print("  ✓ MDS-эмбеддинг работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_augment_features():
    """Тест аугментации признаков эмбеддингами меток."""
    print("\n[TEST] WassersteinEmbedder augment_features")
    try:
        embedder = WassersteinEmbedder(emb_dim=3)
        dataset = MockDataset(num_samples=50, feature_dim=10, num_classes=3)

        label_embeddings = torch.randn(3, 3)
        class_offsets = [0, 3]

        Z = embedder.augment_features(dataset, label_embeddings, 0, class_offsets)

        assert Z.shape == (50, 13)  # 10 features + 3 label embeddings
        print("  ✓ Аугментация признаков работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_compute_wte_single():
    """Тест полного пайплайна WTE для одного датасета."""
    print("\n[TEST] WassersteinEmbedder compute_wte single dataset")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, max_samples=30)
        dataset = MockDataset(num_samples=30, feature_dim=10, num_classes=3)

        task_embs, label_embs, aug_data = embedder.compute_wte([dataset])

        assert task_embs.shape[0] == 1
        assert label_embs.shape == (3, 2)
        assert len(aug_data) == 1
        assert aug_data[0].shape[1] == 12  # 10 + 2
        print("  ✓ WTE для одного датасета работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_compute_wte_multiple():
    """Тест полного пайплайна WTE для нескольких датасетов."""
    print("\n[TEST] WassersteinEmbedder compute_wte multiple datasets")
    try:
        embedder = WassersteinEmbedder(emb_dim=2, max_samples=20)
        ds1 = MockDataset(num_samples=20, feature_dim=8, num_classes=2, seed=1)
        ds2 = MockDataset(num_samples=20, feature_dim=8, num_classes=3, seed=2)

        task_embs, label_embs, aug_data = embedder.compute_wte([ds1, ds2])

        assert task_embs.shape[0] == 2
        assert label_embs.shape == (5, 2)  # 2 + 3 = 5 классов
        assert len(aug_data) == 2
        print("  ✓ WTE для нескольких датасетов работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_clear_cache():
    """Тест очистки кэша."""
    print("\n[TEST] WassersteinEmbedder clear_cache")
    try:
        embedder = WassersteinEmbedder(emb_dim=2)
        dataset = MockDataset(num_samples=20, feature_dim=5, num_classes=2)

        embedder.preprocess_dataset(dataset, dataset_id=0)
        assert len(embedder._data_cache) > 0

        embedder.clear_cache()

        assert len(embedder._data_cache) == 0
        assert len(embedder._stats_cache) == 0
        print("  ✓ Очистка кэша работает корректно")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


def test_wasserstein_text_data():
    """Тест с текстоподобными данными (высокая размерность)."""
    print("\n[TEST] WassersteinEmbedder with text-like data")
    try:
        embedder = WassersteinEmbedder(
            emb_dim=4,
            max_samples=40,
            gaussian_assumption=True,
            diagonal_cov=True
        )

        text_ds = MockVectorizedDataset(
            num_samples=40,
            feature_dim=768,
            num_classes=5,
            seed=1
        )

        X, Y = embedder.preprocess_dataset(text_ds)

        assert X.shape == (40, 768)
        assert Y.shape == (40,)
        print("  ✓ Работа с текстоподобными данными корректна")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Главная функция запуска всех тестов
# ============================================================================

def main():
    """Запуск всех тестов."""
    print("=" * 70)
    print("ЗАПУСК ТЕСТОВ ДЛЯ BaseEmbedder И WassersteinEmbedder")
    print("=" * 70)

    # Список всех тестовых функций
    test_functions = [
        test_base_embedder_init,
        test_base_embedder_device_property,
        test_base_embedder_get_class_statistics,
        test_wasserstein_embedder_init,
        test_wasserstein_embedder_init_custom_params,
        test_wasserstein_preprocess_dataset,
        test_wasserstein_preprocess_dataloader,
        test_wasserstein_preprocess_max_samples,
        test_wasserstein_preprocess_caching,
        test_wasserstein_preprocess_vectorized,
        test_wasserstein_compute_gaussian_stats,
        test_wasserstein_compute_gaussian_stats_diagonal,
        test_wasserstein_bures_distance_identical,
        test_wasserstein_bures_distance_different_means,
        test_wasserstein_pairwise_distances_single,
        test_wasserstein_pairwise_distances_multiple,
        test_wasserstein_embed_distance_matrix,
        test_wasserstein_augment_features,
        test_wasserstein_compute_wte_single,
        test_wasserstein_compute_wte_multiple,
        test_wasserstein_clear_cache,
        test_wasserstein_text_data,
    ]

    # Запуск тестов
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n[ERROR] Неожиданная ошибка в {test_func.__name__}: {e}")
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Итоговый отчёт
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)

    for test_name, result in results:
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        print(f"{status:12} {test_name}")

    print("=" * 70)
    print(f"Всего тестов: {len(results)}")
    print(f"Пройдено:    {passed}")
    print(f"Провалено:   {failed}")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! 🎉")
        return 0
    else:
        print(f"\n⚠️  {failed} ТЕСТ(ОВ) ПРОВАЛЕНО")
        return 1


if __name__ == "__main__":
    exit_code = main()
