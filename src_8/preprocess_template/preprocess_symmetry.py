from mimetypes import init
import os

dir = "data/templates/sphere96"
vertices_path = os.path.join(dir, "vertices.txt")

processed_vertices = []
init_params = []
for i, l in enumerate(open(vertices_path, 'r')):
    value = l.strip().split(' ')
    if value[0] == 'RegularVertex':
        _, a, b, c = value
        init_params.append([float(a), float(b), float(c)])
        processed_vertices.append(i)

f = open(os.path.join(dir, 'symmetries.txt'), 'w')
processed = []
for idx in processed_vertices:
    if idx in processed:
        continue
    vertice = init_params[idx]
    idx2 = init_params.index([-vertice[0], vertice[1], vertice[2]])
    if idx == idx2:
        continue
    processed.append(idx)
    processed.append(idx2)
    f.write(f"{idx} {idx2}\n")
