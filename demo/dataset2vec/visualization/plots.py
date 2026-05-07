from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class EmbeddingVisualizer:
    """Универсальная визуализация эмбеддингов.

    Поддерживает наложение уменьшения размерности через PCA/TSNE и построение
    2D/3D scatter plot.

    Пример:
        vis = EmbeddingVisualizer(random_state=42)
        fig, ax = vis.plot(embeddings, labels=labels, method="tsne")
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def reduce(
        self,
        embeddings: NDArray[np.floating],
        method: str = "tsne",
        n_components: int = 2,
        **kwargs,
    ) -> NDArray[np.floating]:
        """Уменьшает размерность эмбеддингов (PCA/TSNE)."""
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (n_samples, n_features)")
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 или 3")

        # Если эмбеддинги уже нужной размерности, ничего не делаем.
        if embeddings.shape[1] == n_components:
            return embeddings

        method = method.lower()
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=self.random_state, **kwargs)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=self.random_state, **kwargs)
        else:
            raise ValueError("Метод должен быть 'tsne' или 'pca'")

        return reducer.fit_transform(embeddings)

    def plot(
        self,
        embeddings: NDArray[np.floating],
        labels: list[str] | None = None,
        title: str | None = None,
        method: str = "tsne",
        n_components: int = 2,
        annotate: bool = False,
        figsize: tuple[int, int] = (6, 6),
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Построить scatter plot эмбеддингов.

        Args:
            embeddings: Массив shape=(n_samples, n_features).
            labels: Метки для подписи точек (опционально).
            method: "tsne" или "pca".
            n_components: 2 или 3.
            annotate: показывает подписи (labels) рядом с точками.
            show: вызывает plt.show() (по умолчанию True).
            **kwargs: дополнительные параметры для TSNE/PCA.

        Возвращает:
            (fig, ax) matplotlib для дальнейшей кастомизации.
        """
        transformed = self.reduce(
            embeddings, method=method, n_components=n_components, **kwargs
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Цвета по меткам (если есть) или единый цвет
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=str)
            unique_labels = list(dict.fromkeys(labels_arr.tolist()))
            cmap = plt.get_cmap("tab10")
            colors = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(unique_labels)}

        if n_components == 2:
            x, y = transformed[:, 0], transformed[:, 1]

            if labels is not None:
                for lbl in unique_labels:
                    mask = labels_arr == lbl
                    ax.scatter(
                        x[mask], y[mask],
                        c=[colors[lbl]],
                        alpha=0.7,
                        label=lbl,
                        edgecolors="none",
                    )
                if not annotate:
                    ax.legend(title="label")
            else:
                ax.scatter(x, y, c="tab:blue", alpha=0.7)

            if labels is not None and annotate:
                for i, label in enumerate(labels_arr):
                    ax.text(x[i], y[i], label, fontsize=8)

            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
        else:
            from mpl_toolkits.mplot3d import Axes3D

            ax = fig.add_subplot(111, projection="3d")
            x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]

            if labels is not None:
                for lbl in unique_labels:
                    mask = labels_arr == lbl
                    ax.scatter(
                        x[mask], y[mask], z[mask],
                        c=[colors[lbl]],
                        alpha=0.7,
                        label=lbl,
                        edgecolors="none",
                    )
                if not annotate:
                    ax.legend(title="label")
            else:
                ax.scatter(x, y, z, c="tab:blue", alpha=0.7)

            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
            ax.set_zlabel("dim 2")

        if title is not None:
            ax.set_title(title)
        ax.grid(True)
        if show:
            plt.show()
        return fig, ax


def plot_embeddings(
    embeddings: NDArray[np.floating],
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Простейший scatter plot эмбеддингов."""
    if embeddings.shape[1] < 2:
        raise ValueError("Embeddings must have at least 2 dimensions for plotting")

    x = embeddings[:, 0]
    y = embeddings[:, 1]

    plt.figure(figsize=(6, 6))

    if labels is not None:
        labels_arr = np.asarray(labels, dtype=str)
        unique_labels = list(dict.fromkeys(labels_arr.tolist()))
        cmap = plt.get_cmap("tab10")
        colors = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(unique_labels)}
        for lbl in unique_labels:
            mask = labels_arr == lbl
            plt.scatter(x[mask], y[mask], c=[colors[lbl]], alpha=0.7, label=lbl, edgecolors="none")
        plt.legend(title="label")
    else:
        plt.scatter(x, y, c="tab:blue", alpha=0.7)

    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(x[i], y[i], label, fontsize=8)
    if title is not None:
        plt.title(title)
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.grid(True)
    plt.show()