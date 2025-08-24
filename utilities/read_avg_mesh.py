import numpy as np
import nibabel as nib
from pathlib import Path
import torch

def load_surf_and_label(fs_dir, subject, hemi, surf_type='inflated'):
    
    lr = '.L.' if hemi == 'lh' else '.R.'
    surf_path = fs_dir / f"{subject}{lr}{surf_type}.32k_fs_LR.surf.gii"
    label_path = fs_dir / f"{subject}{lr}aparc.32k_fs_LR.label.gii"
    surf_gii = nib.load(str(surf_path))
    label_gii = nib.load(str(label_path))
    vertices = np.array(surf_gii.darrays[0].data)  # (32492, 3)
    faces = np.array(surf_gii.darrays[1].data)     # (n_faces, 3)
    mars_label = np.array(label_gii.darrays[0].data)
    return vertices, faces, mars_label

def compute_vertex_neighbors(faces, num_vertices):
    neighbors = {i: set() for i in range(num_vertices)}
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i+1)%3]
            v3 = face[(i+2)%3]
            neighbors[v1].update([v2, v3])
    neighbors = {i: np.array(list(neigh)) for i, neigh in neighbors.items()}
    return neighbors

def compute_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    return face_areas

def CBIG_AvgMesh_across_subjects(base_dir, hemi, surf_type='inflated', label_type='aparc'):
    """
    Compute the average mesh (vertex coordinates) across all subjects for a hemisphere.
    Returns avg_mesh dict compatible with your code.
    """
    subjects = [d for d in Path(base_dir).iterdir() if d.is_dir() and (d / "hcp1200" / "fsaverage_LR32k").exists()]
    all_vertices = []
    all_mars_labels = []
    faces_ref = None

    for subj in subjects:
        fs_dir = subj / "hcp1200" / "fsaverage_LR32k"
        subject_id = subj.name
        try:
            vertices, faces, mars_label = load_surf_and_label(fs_dir, subject_id, hemi, surf_type)
        except Exception as e:
            print(f"Skipping {subject_id}: {e}")
            continue
        if faces_ref is None:
            faces_ref = faces
            mars_label_ref = mars_label
        else:
            # Check faces/topology match
            if not np.array_equal(faces, faces_ref):
                raise ValueError(f"Mesh faces for {subject_id} do not match reference mesh!")
            # Check medial wall mask match
            if not np.array_equal(mars_label, mars_label_ref):
                print(f"Warning: Medial wall mask mismatch in {subject_id} (will use reference mask).")
        all_vertices.append(vertices)
        all_mars_labels.append(mars_label)

    if not all_vertices:
        raise RuntimeError("No subject meshes loaded!")

    all_vertices = np.stack(all_vertices, axis=0)
    avg_vertices = np.mean(all_vertices, axis=0)
    avg_vertices = torch.tensor(avg_vertices, dtype=torch.float32)
    faces = torch.tensor(faces_ref, dtype=torch.long)
    mars_label = torch.tensor(mars_label_ref, dtype=torch.int32)
    num_vertices = avg_vertices.shape[0]
    neighbors_dict = compute_vertex_neighbors(faces, num_vertices)
    vertex_nbors = [neighbors_dict[i] for i in range(num_vertices)]
    faceAreas = compute_face_areas(avg_vertices.numpy(), faces.numpy())

    mesh_data = torch.arange(num_vertices)
    faces_vis = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    try:
        import pyvista as pv
        vis_mesh_data = pv.PolyData(np.array(avg_vertices), faces_vis)
    except ImportError:
        vis_mesh_data = None

    avg_mesh = {
        'MARS_label': mars_label,
        'vertices': avg_vertices,
        'faces': faces,
        'vertexNbors': vertex_nbors,
        'mesh_data': mesh_data,
        'vis_mesh_data': vis_mesh_data,
        'faceAreas': faceAreas,
        # (add any other fields you want)
    }
    return avg_mesh


#####################################################################################################     

def extract_submesh(vertices, faces, cortex_mask):
    """Return submesh (vertices, faces) containing only cortex_mask vertices."""
    # Map old vertex indices to new indices
    old2new = -np.ones(len(cortex_mask), dtype=int)
    old2new[np.where(cortex_mask)[0]] = np.arange(np.sum(cortex_mask))
    # Keep faces where all 3 vertices are in cortex
    keep_faces = np.all(cortex_mask[faces], axis=1)
    faces_sub = faces[keep_faces]
    # Re-index faces
    faces_sub = old2new[faces_sub]
    vertices_sub = vertices[cortex_mask]
    return vertices_sub, faces_sub, old2new

def compute_vertex_neighbors_sub(faces, num_vertices):
    """Neighbors for a submesh, with n_vertices = len(vertices_sub)"""
    neighbors = {i: set() for i in range(num_vertices)}
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i+1)%3]
            v3 = face[(i+2)%3]
            neighbors[v1].update([v2, v3])
    neighbors = {i: np.array(list(neigh)) for i, neigh in neighbors.items()}
    return neighbors
def CBIG_AvgMesh_intersection(base_dir, hemi, surf_type='inflated', label_type='aparc'):
    """
    Compute average mesh and intersection cortex mask, restrict mesh to intersection cortex only.
    Returns avg_mesh dict.
    """
    subjects = [d for d in Path(base_dir).iterdir() if d.is_dir() and (d / "hcp1200" / "fsaverage_LR32k").exists()]
    all_vertices = []
    all_mars_labels = []
    faces_ref = None

    for subj in subjects:
        fs_dir = subj / "hcp1200" / "fsaverage_LR32k"
        subject_id = subj.name
        
        try:
            vertices, faces, mars_label = load_surf_and_label(fs_dir, subject_id, hemi, surf_type)
            
        except Exception as e:
            print(f"Skipping {subject_id}: {e}")
            continue
        if faces_ref is None:
            faces_ref = faces
        else:
            if not np.array_equal(faces, faces_ref):
                raise ValueError(f"Mesh faces for {subject_id} do not match reference mesh!")
        all_vertices.append(vertices)
        all_mars_labels.append(mars_label)
    
    if not all_vertices:
        raise RuntimeError("No subject meshes loaded!")
    
    all_vertices = np.stack(all_vertices, axis=0)      # (n_subjects, n_vertices, 3)
    all_mars_labels = np.stack(all_mars_labels, axis=0)# (n_subjects, n_vertices)
    
    # 1. Compute intersection cortex mask
    cortex_mask = np.all(all_mars_labels != -1, axis=0)  # (n_vertices,)

    # 2. Average only for intersection vertices
    avg_vertices = np.mean(all_vertices[:, cortex_mask, :], axis=0)
    faces_sub = None
    vertices_sub = None

    # 3. Build submesh for intersection cortex
    vertices_sub, faces_sub, old2new = extract_submesh(all_vertices[0], faces_ref, cortex_mask)
    num_vertices_sub = vertices_sub.shape[0]

    # 4. Compute neighbors for submesh
    neighbors_dict = compute_vertex_neighbors_sub(faces_sub, num_vertices_sub)
    vertex_nbors = [neighbors_dict[i] for i in range(num_vertices_sub)]
    faceAreas = compute_face_areas(vertices_sub, faces_sub)

    mesh_data = torch.arange(num_vertices_sub)
    try:
        import pyvista as pv
        faces_vis = np.hstack([np.full((faces_sub.shape[0], 1), 3), faces_sub])
        vis_mesh_data = pv.PolyData(np.array(vertices_sub), faces_vis)
    except ImportError:
        vis_mesh_data = None
    
    avg_mesh = {
        'MARS_label': cortex_mask.astype(np.int32),   # Now a boolean mask: 1=cortex, 0=medial wall
        'vertices': torch.tensor(avg_vertices, dtype=torch.float32),
        'faces': torch.tensor(faces_sub, dtype=torch.long),
        'vertexNbors': vertex_nbors,
        'mesh_data': mesh_data,
        'vis_mesh_data': vis_mesh_data,
        'faceAreas': faceAreas,
        'cortex_indices': np.where(cortex_mask)[0],   # Save which vertices are cortex
    }
    return avg_mesh

def build_masked_vertex_nbors(faces, mask):
    # mask: boolean array (True = keep)
    # faces: (n_faces, 3)
    idx_map = -np.ones(len(mask), dtype=int)
    idx_map[np.where(mask)[0]] = np.arange(np.sum(mask))
    kept_faces = []
    for f in faces:
        if all(mask[f]):
            kept_faces.append(idx_map[f])
    kept_faces = np.array(kept_faces)
    n_valid = np.sum(mask)
    neighbors = {i: set() for i in range(n_valid)}
    for face in kept_faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i+1)%3]
            v3 = face[(i+2)%3]
            neighbors[v1].update([v2, v3])
    return [np.array(list(neighbors[i])) for i in range(n_valid)]