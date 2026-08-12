"""Mesh construction utilities for the image-to-STL stage.

Core subdivision, smoothing, and back-head reconstruction logic is preserved
from the original ``back_head_v6_2.py`` project code. Persistent intermediate
vertices/faces CSV files are intentionally removed from the reconstructed
pipeline.
"""

from pathlib import Path

import numpy as np
from stl import mesh


def subdivide_triangle_with_continuity(v1, v2, v3, resolution, vertices_dict=None):
    """
    삼각형을 연속성을 유지하며 subdivision.->기존의 삼각형을 더 작은 삼각형들로 나누되, 나뉜 삼각형들 사이의 경계가 매끄럽게 이어지도록 하는 과정
    """
    if vertices_dict is None:
        vertices_dict = {}

    def get_or_add_vertex(vertex):
        vertex_tuple = tuple(np.round(vertex, decimals=8))
        if vertex_tuple not in vertices_dict:
            vertices_dict[vertex_tuple] = len(vertices_dict)
        return vertices_dict[vertex_tuple]

    points = []
    index_map = []

    for i in range(resolution + 1):
        row_indices = []
        for j in range(resolution + 1 - i):
            w1 = i / resolution
            w2 = j / resolution
            w3 = 1 - w1 - w2
            point = w1 * v1 + w2 * v2 + w3 * v3
            index = get_or_add_vertex(point)
            row_indices.append(index)
        index_map.append(row_indices)

    triangles = []
    for i in range(resolution):
        for j in range(resolution - i):
            p1 = index_map[i][j]
            p2 = index_map[i + 1][j]
            p3 = index_map[i][j + 1]
            triangles.append([p1, p2, p3])
            if j < resolution - i - 1:
                p4 = index_map[i + 1][j + 1]
                triangles.append([p2, p4, p3])

    return points, triangles, vertices_dict


def hc_laplacian_smoothing(vertices, faces, iterations, alpha=0.3, beta=0.5):
    """
    HC (Humphrey's Classes) Laplacian Smoothing 적용.
    alpha: 원래 위치에 대한 가중치 (0~1)
    beta: 역보정 강도 (0~1)
    """
    vertices = np.array(vertices, dtype=float)
    original_vertices = vertices.copy()

    adjacency_list = {i: [] for i in range(len(vertices))}
    for face in faces:
        for i in range(3):
            adjacency_list[face[i]].extend(face[j] for j in range(3) if j != i)
    for key in adjacency_list:
        adjacency_list[key] = list(set(adjacency_list[key]))

    for _ in range(iterations):
        q = vertices.copy()
        for i, neighbors in adjacency_list.items():
            if len(neighbors) > 0:
                avg_position = np.mean(vertices[neighbors], axis=0)
                q[i] = avg_position

        b = np.zeros_like(vertices)
        for i in range(len(vertices)):
            b[i] = q[i] - (alpha * original_vertices[i] + (1 - alpha) * vertices[i])

        new_vertices = q.copy()
        for i, neighbors in adjacency_list.items():
            if len(neighbors) > 0:
                avg_b = np.mean(b[neighbors], axis=0)
                new_vertices[i] = q[i] - (beta * b[i] + (1 - beta) * avg_b)

        vertices = new_vertices

    return vertices


def find_boundary_edges(faces):
    """
    삼각형 목록에서 한 번만 등장하는 edge를 boundary edge로 판정.
    """
    edge_count = {}

    for face in faces:
        edges = [
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0])
        ]

        for e in edges:
            e = tuple(sorted(e))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    return boundary_edges


def order_boundary_loop(boundary_edges):
    """
    boundary edge들을 하나의 순서 있는 loop로 정렬.
    현재 얼굴 외곽선처럼 하나의 큰 폐곡선이라고 가정.
    """
    if not boundary_edges:
        raise ValueError("No boundary edges found in the face mesh.")

    adjacency = {}

    for v1, v2 in boundary_edges:
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    start = boundary_edges[0][0]
    loop = [start]

    prev = None
    current = start

    while True:
        neighbors = adjacency[current]
        next_candidates = [n for n in neighbors if n != prev]

        if not next_candidates:
            break

        next_v = next_candidates[0]

        if next_v == start:
            break

        loop.append(next_v)

        prev = current
        current = next_v

        if len(loop) > len(boundary_edges) + 5:
            break

    return loop


def create_back_head_mesh(
    vertices,
    faces,
    n_rings=12,
    back_depth_ratio=0.6,
    shrink_power=1.35,
    top_lift_ratio=0.32,
    crown_back_ratio=0.42,
    upper_occipital_ratio=0.22,
    upper_occipital_back_ratio=0.4,
    upper_occipital_width=0.7,
    smooth_top=True
):
    """
    얼굴 외곽 boundary에서 뒤쪽으로 contour ring을 만들되,
    위쪽 머리뚜껑과 후두부 상부 볼륨까지 생성하는 함수.
    (원본 코드 100% 그대로)
    """

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)

    boundary_edges = find_boundary_edges(faces)
    boundary_loop = order_boundary_loop(boundary_edges)
    boundary_pts = vertices[boundary_loop]

    center = np.mean(boundary_pts, axis=0)

    x_min, x_max = np.min(boundary_pts[:, 0]), np.max(boundary_pts[:, 0])
    y_min, y_max = np.min(boundary_pts[:, 1]), np.max(boundary_pts[:, 1])
    z_min, z_max = np.min(boundary_pts[:, 2]), np.max(boundary_pts[:, 2])

    width = x_max - x_min
    height = y_max - y_min
    face_size = max(width, height)

    x_center = 0.5 * (x_min + x_max)

    # MediaPipe z 방향이 반대로 나오면 여기만 -1.0으로 바꾸면 됨
    back_dir = np.array([0.0, 0.0, 1.0])

    back_depth = face_size * back_depth_ratio
    top_lift = height * top_lift_ratio
    crown_back = back_depth * crown_back_ratio

    new_vertices = vertices.tolist()
    new_faces = faces.tolist()

    ring_indices = []
    ring_indices.append(boundary_loop)

    n = len(boundary_loop)

    # ── 원본 첫 번째 vertex index 기록 (후처리 smoothing 범위 지정용) ──
    back_head_start_idx = len(new_vertices)

    for r in range(1, n_rings + 1):
        t = r / n_rings

        scale = np.cos(t * np.pi / 2) ** shrink_power
        posterior_weight = np.sin(t * np.pi / 2)

        z_offset = back_dir * back_depth * posterior_weight

        current_ring = []

        for p in boundary_pts:
            y_norm = (p[1] - y_min) / (height + 1e-8)

            top_weight = np.clip(1.0 - y_norm, 0.0, 1.0)
            top_weight = top_weight ** 0.65

            radial = p - center
            new_p = center + radial * scale + z_offset

            if smooth_top:
                crown_weight = np.exp(
                    -((back_depth * posterior_weight - crown_back) ** 2) /
                    (2 * (0.45 * back_depth) ** 2)
                )

                lift = top_lift * top_weight * (
                    0.45 * posterior_weight + 0.85 * crown_weight
                )

                new_p[1] -= lift

                # 기존 정수리/상부 z 보정
                new_p += back_dir * back_depth * 0.12 * top_weight * crown_weight

                # ----------------------------------------------------
                # 추가: 후두부 상부 볼륨 보강
                # ----------------------------------------------------
                x_rel = (new_p[0] - x_center) / (0.5 * width + 1e-8)

                center_weight = np.exp(
                    -(x_rel ** 2) / (2 * upper_occipital_width ** 2)
                )

                occipital_weight = np.exp(
                    -((posterior_weight - upper_occipital_back_ratio) ** 2) /
                    (2 * 0.18 ** 2)
                )

                upper_weight = top_weight ** 1.2

                occipital_bulge = (
                    upper_occipital_ratio
                    * height
                    * upper_weight
                    * center_weight
                    * occipital_weight
                )

                # 위로 채우기
                new_p[1] -= occipital_bulge

                # 뒤로도 같이 밀어서 후두부 상부 볼륨처럼 보이게 함
                new_p += back_dir * occipital_bulge * 0.65

            new_vertices.append(new_p.tolist())
            current_ring.append(len(new_vertices) - 1)

        ring_indices.append(current_ring)

    # ring 사이 삼각형 strip 연결
    for r in range(len(ring_indices) - 1):
        ring_a = ring_indices[r]
        ring_b = ring_indices[r + 1]

        for i in range(n):
            a0 = ring_a[i]
            a1 = ring_a[(i + 1) % n]
            b0 = ring_b[i]
            b1 = ring_b[(i + 1) % n]

            new_faces.append([a0, b0, a1])
            new_faces.append([a1, b0, b1])

    # 마지막 ring을 중심점으로 cap 처리
    last_ring = ring_indices[-1]
    last_pts = np.array([new_vertices[idx] for idx in last_ring])
    cap_center = np.mean(last_pts, axis=0)

    cap_idx = len(new_vertices)
    new_vertices.append(cap_center.tolist())

    for i in range(n):
        a = last_ring[i]
        b = last_ring[(i + 1) % n]
        new_faces.append([a, cap_idx, b])

    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces, dtype=int)

    back_head_indices = set(range(back_head_start_idx, len(new_vertices)))

    # adjacency 구축 (뒷통수 face만)
    adjacency = {i: set() for i in back_head_indices}
    for face in new_faces:
        for i in range(3):
            vi = face[i]
            if vi in back_head_indices:
                for j in range(3):
                    if i != j and face[j] in back_head_indices:
                        adjacency[vi].add(face[j])

    # Laplacian smoothing 반복
    smooth_iterations = 5
    smooth_lambda = 0.9

    for _ in range(smooth_iterations):
        smoothed = new_vertices.copy()
        for vi in back_head_indices:
            neighbors = adjacency.get(vi, set())
            if len(neighbors) == 0:
                continue
            avg = np.mean(new_vertices[list(neighbors)], axis=0)
            smoothed[vi] = (1 - smooth_lambda) * new_vertices[vi] + smooth_lambda * avg
        new_vertices = smoothed

    return new_vertices, new_faces


def build_smoothed_face_mesh(
    landmarks,
    mesh_index,
    resolution=3,
    iterations=10,
    alpha=0.3,
    beta=0.5,
):
    """Subdivide the MediaPipe face topology and apply HC Laplacian smoothing."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1")

    landmarks = np.asarray(landmarks, dtype=float)
    mesh_index = np.asarray(mesh_index, dtype=int)

    subdivided_vertices_dict = {}
    subdivided_faces = []

    for triangle in mesh_index:
        v1, v2, v3 = landmarks[triangle]
        _, local_faces, subdivided_vertices_dict = subdivide_triangle_with_continuity(
            v1,
            v2,
            v3,
            resolution=resolution,
            vertices_dict=subdivided_vertices_dict,
        )
        subdivided_faces.extend(local_faces)

    subdivided_vertices = np.array(
        list(subdivided_vertices_dict.keys()),
        dtype=float,
    )
    subdivided_faces = np.asarray(subdivided_faces, dtype=int)

    smoothed_vertices = hc_laplacian_smoothing(
        subdivided_vertices,
        subdivided_faces,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
    )

    return smoothed_vertices, subdivided_faces


def close_face_mesh(vertices, faces):
    """Add the reconstructed back head using the original v6_2 parameters."""
    return create_back_head_mesh(
        vertices,
        faces,
        n_rings=36,
        back_depth_ratio=0.7,
        shrink_power=1.2,
        top_lift_ratio=0.4,
        crown_back_ratio=0.5,
        upper_occipital_ratio=0.13,
        upper_occipital_back_ratio=0.85,
        upper_occipital_width=0.85,
        smooth_top=True,
    )


def save_stl(vertices, faces, output_path):
    """Write the closed mesh to STL using the original y/z orientation flips."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=float).copy()
    faces = np.asarray(faces, dtype=int)

    # Preserve the coordinate-orientation conversion used by back_head_v6_2.py.
    vertices[:, 1] *= -1
    vertices[:, 2] *= -1

    stl_faces = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        for j in range(3):
            stl_faces["vectors"][i][j] = vertices[face[j]]

    face_mesh = mesh.Mesh(data=stl_faces)
    face_mesh.save(str(output_path))
