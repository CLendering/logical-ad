import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random


def visualize_scene_graph(image, scene_graph, violations=None, show=True):
    """
    Visualizes the Neuro-Symbolic Scene Graph overlaying the original image.

    Args:
        image (np.ndarray): The original RGB image.
        scene_graph (SceneGraph): The graph output from NS-AD.
        violations (List[Violation]): Optional list of anomalies to highlight.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    # Color map for Prototype IDs (Consistent colors for same types)
    unique_protos = set(n.prototype_id for n in scene_graph.nodes)
    random.seed(42)
    colors = {
        pid: (random.random(), random.random(), random.random())
        for pid in unique_protos
    }

    # 1. Draw Edges (Relationships)
    node_map = {n.id: n for n in scene_graph.nodes}

    if hasattr(scene_graph, "edges"):
        for src_id, neighbors in scene_graph.edges.items():
            if src_id not in node_map:
                continue
            src_node = node_map[src_id]

            for dst_id, _, _ in neighbors:
                if dst_id not in node_map:
                    continue
                dst_node = node_map[dst_id]

                ax.arrow(
                    src_node.centroid[0],
                    src_node.centroid[1],
                    dst_node.centroid[0] - src_node.centroid[0],
                    dst_node.centroid[1] - src_node.centroid[1],
                    color="white",
                    alpha=0.3,
                    head_width=5,
                    length_includes_head=True,
                )

    # 2. Draw Nodes (Objects)
    for node in scene_graph.nodes:
        x1, y1, x2, y2 = node.bbox
        w, h = x2 - x1, y2 - y1
        c = colors.get(node.prototype_id, (1, 0, 0))

        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor=c, facecolor="none"
        )
        ax.add_patch(rect)

        label = f"ID:{node.id}\nType:{node.prototype_id}"
        ax.text(x1, y1 - 5, label, color="white", fontsize=8, backgroundcolor=c)
        ax.plot(node.centroid[0], node.centroid[1], "o", color="yellow", markersize=4)

    # 3. Highlight Violations
    title_color = "black"
    if violations:
        title_color = "red"
        text_str = "ANOMALIES DETECTED:\n" + "\n".join(
            [f"- {v.description}" for v in violations[:5]]
        )
        if len(violations) > 5:
            text_str += f"\n...and {len(violations)-5} more."

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            color="red",
        )

    ax.set_title(
        f"Neuro-Symbolic Scene Graph: {scene_graph.image_id}",
        color=title_color,
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    if show:
        plt.show()

    return fig
