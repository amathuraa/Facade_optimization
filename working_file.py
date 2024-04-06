from datetime import datetime
import numpy as np
from numba import jit
import cProfile
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go


## TASKS:
# VECTORIZE FOR LOOPS:
# a) Reduce all triangles dimensions X
# b) Calculate best_fit_trasnform X output: rotations,translations
# c) Rotate and translate reduced_triangles using T. X X Output: rotated_triangles
# d) Calculate distance matrix using rotated_triangles O
# e) Cluster triangles based on distance matrix and find mean O
# f) Send mean back to original triangle position O


def rotation_matrix_to_x_axis(vector):
    # Normalize vector
    normalized_vector = vector / np.linalg.norm(vector)
    
    # Calculate rotation axis
    rotation_axis = np.cross(normalized_vector, np.array([1, 0, 0]))
    
    if np.linalg.norm(rotation_axis) != 0:
        rotation_axis /= np.linalg.norm(rotation_axis)

    
    # Calculate angle between vector and x-axis
    dot_product = np.dot(normalized_vector, np.array([1, 0, 0]))
    angle_rad = np.arccos(dot_product)
    
    # Calculate rotation matrix
    skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = (np.cos(angle_rad) * np.eye(3) +
                       (1 - np.cos(angle_rad)) * np.outer(rotation_axis, rotation_axis) +
                       np.sin(angle_rad) * skew_matrix)
    
    return rotation_matrix


def rotation_matrix_to_xy_plane(points):
    # Normalize vector
    #normalized_vector1 = vector1 / np.linalg.norm(vector1)
    #normalized_vector2= vector2 / np.linalg.norm(vector2)
    vector1 = points[1]
    vector2 = points[2]
    normal = np.cross(vector1,vector2)
    normal_unit = normal / np.linalg.norm(normal)
    angle = np.arccos(np.dot(normal_unit,[0,0,1]))
    
    rotation_axis = np.cross(normal_unit,[0,0,1])
   
    #Rodrigues
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate the cross product matrix
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    # Calculate the rotation matrix
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Perform the rotation.
    rotated_points = np.dot(rotation_matrix, np.array(points).T).T
    
    return rotated_points, rotation_matrix


def rotate_points(rotation_matrix, points):
    rotated_points = []
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_points.append(list(rotated_point))
    return rotated_points


def originate(triangle):
    xi, yi, zi = triangle[0]
    originated_triangle=[(x - xi,y - yi,z - zi) for x, y, z in triangle]
    return originated_triangle, (xi, yi, zi)


def reduce_dimensions(source_points):
    source,translation = originate(source_points)
    base_vector = source[1]
    
    rotation_matrix = rotation_matrix_to_x_axis(base_vector)
    rotated_triangle = rotate_points(rotation_matrix, np.array(source))
    
    plane_vector = rotated_triangle[2]
    vector2 = [x - y for x, y in zip(plane_vector,rotated_triangle[1])]
    rotated_triangle_flat, xy_matrix = rotation_matrix_to_xy_plane(rotated_triangle)
    z_vals = [row[2] for row in rotated_triangle_flat]
    
    assert max(z_vals) <= 1e-6, "Z Values are not 0"
    
    source_rotated = [[row[0], row[1]] for row in rotated_triangle_flat]
    rot_matrix_combined = np.dot(xy_matrix,rotation_matrix)
    #print(type(source_rotated))
    return source_rotated, [translation, rot_matrix_combined]


# TODO:
# Separation of concerns -> reads data, cleans data and also calculates triangles (can separate into diff funcs)
def clean_csv(filename,lenght=0):
    col_names = ['UID','First','Second','Third']
    data = pd.read_csv(filename, names=col_names,skiprows=1)
    columns = {}
    df = pd.DataFrame(columns=[0])
    col = []
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        ID=row[0]
        values1 = row[1].strip('{}').split(',')
        values2 = row[2].strip('{}').split(',')
        values3 = row[3].strip('{}').split(',')
        df.loc[ID]=[np.array([[values1[0],values1[1],values1[2]],[values2[0],values2[1],values2[2]],[values3[0],values3[1],values3[2]]]).astype(float)]
    orig_info = {}
    p = len(df)
    UID_names = df.index
    tris = []
    for i in range(p):
        reduced, info = reduce_dimensions(df.loc[UID_names[i]][0])
        tris.append(reduced)
        orig_info[UID_names[i]]= info
    triangles = np.array(tris)
    if lenght:
        triangles=triangles[:lenght]
    print(triangles)
    print('TYPE:',type(triangles),triangles.shape)
    return triangles, orig_info

# TODO:
# permutation loop should at least use relevant itertools helper function
# any way to avoid the special reflection case? ignore it / precompute it / postcompute
# rots and trans should probably be preallocated ndarrays
#@jit

def vectorized_test(A):
    print(A.shape,A[0].shape)
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(A, axis=1)
    
    AA = A - centroid_A[:, np.newaxis, :]
    BB = A - centroid_B[:, np.newaxis, :]
    print('Calcs:',A,centroid_B.shape,centroid_A[0].shape,BB.shape)
    Rots=np.empty((len(A)**2,2,2))
    trans=np.empty((len(A)**2,1,2))
    m = AA[0].shape[1]
    
    #ADD LOOP FOR PERMUTATIONS
    for i in range(len(A)):
        for j in range(len(A)):
            H = np.dot(AA[i].T, BB[j])
            
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # Special reflection case
            if np.linalg.det(R) < 0:
                print('det')
                Vt[m-1,:] *= -1
                R = np.dot(Vt.T, U.T)
            t = centroid_B[j].reshape(-1,1) - np.dot(R, centroid_A[i].reshape(-1,1))
            
            Rots[i*len(A)+j]=R
            trans[i*len(A)+j]=t.T
    return Rots,trans


def rotate_triangles(rotation_list, translation_list, triangles):
    
    n = int(np.sqrt(rotation_list.shape[0]))
    
    rotation_matrices = rotation_list.reshape(n, n, 2, 2)

    triangles_transposed = np.transpose(triangles, (0, 2, 1))
    # Using np.einsum to perform the rotation operations
    result = np.einsum('ijkl,jlm->ijkm', rotation_matrices, triangles_transposed)

    # Transpose back to the original shape (3, 3, 3, 2)
    result = np.transpose(result, (0, 1, 3, 2))
    translation_matrix = translation_list.reshape(n,n,1,2)
    
    rotated_triangles = result + translation_matrix
    
    return rotated_triangles


def distance_matrix_calc(rotated_triangles, triangles):
    diffcalc = rotated_triangles - triangles[:,np.newaxis]
    diff_square = np.square(diffcalc)
    diff_sum_ = np.sum(diff_square, axis=(2))
    distance_matrix = np.sum(diff_sum_, axis=(2)) / 3
    return distance_matrix
    

def cluster(distance_matrix, final_tris, k):
    def graph(fig, triangle1, name='unnamed', color='red', orient=[0,1,2], opacity=0.4):
        x=[row[0] for row in triangle1]
        y=[row[1] for row in triangle1]
        #z=[row[2] for row in triangle1]
        if len(triangle1[0]) == 3:
            z = [row[2] for row in triangle1]
        else:
            z = np.array([0 for x in range(len(triangle1))])
        RGB = [ 'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 256)','rgb(255,255,255)']
        vertex_col = [RGB[x] for x in orient]
        
        fig.add_trace(go.Mesh3d(x=np.array(x),
                                y=np.array(y),
                                z=np.array(z),
                                color=color,
                                opacity=opacity,
                                name=name,))
        #print(x,y,z)
        #fig.show()
        #print(orient)

    means={}
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=k, linkage='complete').fit(distance_matrix)

    labels=model.labels_
    print(labels)

    clusters = {}

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    model_error=0
    for i in range(k):
        print(f'Cluster {i}:')
        avg = 0
        best_fit = float('inf')
        #print('Average error per point:')
        for j in clusters[i]:
            distance_list = [distance_matrix[j,x] for x in clusters[i]]
            #print(distance_list)
            sums = sum(distance_list)
            av = sums / (len(clusters[i]))
            #print(av)
            avg = avg+av
            if av < best_fit:
                best_fit = av
                best_tri = j

        #QUESTION: SHould best tri be included in mean? (if x!=best_tri)
        base_tris = final_tris[best_tri][[x for x in clusters[i]]]
        
        clust_err = avg / (len(clusters[i]))
        model_error = model_error + clust_err
        #print(clusters[i])
        print('Average error within cluster:', clust_err)
        print('best_tri:', best_tri)
        
        #print(base_tris)
        
        fig=go.Figure()
        
        if len(clusters[i]) > 1:
            mean = np.mean(base_tris, axis=0)
            print('Triangle Centroid', mean)
            graph(fig, np.array(mean), name='Mean', color='black')
            
        else: 
            mean = final_tris[best_tri][best_tri]
        #graph(fig,np.array(final_tris[best_tri][best_tri]),name='best_tri')
        for m in range(len(base_tris)):
            graph(fig, np.array(base_tris[m]), name=m)
       
        # anything wrong with { } notation for dictionaries? ;)
        fig.update_traces(showlegend=True, selector=dict(type='mesh3d'))
        camera = dict(
            eye = dict(x=0., y=0., z=2.5)
            )
        name_ICP = f'Cluster {i}'
        fig.update_layout(scene_camera=camera, 
                          title=name_ICP,
                          scene_aspectmode='cube')

        fig.show()
        means[i] = (mean, best_tri, clusters[i])
    
    model_error = model_error / k
    print('Model Error:', model_error)
    
    return model_error, means
## Code responsible for sending the mean triangles back to replace the original ones NEEDS TO BE VECTORIZED
def find_X(XABC,orig_info):
    '''
    XABC: Centroid of the cluster
    A: 3x3 Matrix responsible for rotation to the x axis
    B: 3x3 matrix responsible for rotation to the XY plane
    C: 2x2 matrix responsible for ICP rotation
    t: (1,3) responsible for ICP Translation
    per:[1,3] resposible for ordering of points post ICP 
    origin_info: (1,3) responsible for originiating triangle)
    
    OUTPUT: 
    X: The original triangles position in space
    
    USAGE: 
    Given the centroid of a cluster, this function can project it into the poisitions of the other triangles within the cluster
    ''' 
    
    #A = orig_info[0][1].T
    #B = orig_info[0][2].T
    AB=orig_info[0][1].T
    C = orig_info[1][0:-1, 0:-1]
    #C=T_2[:2, :2]
    t = orig_info[1][:-1, -1]
    #t=T_2[:-1, -1]
    O = [-x for x in orig_info[0][0]]
    
    

    # X_o_ABC_t = np.matmul(C.T,XABC)
    X_o_AB = (np.dot(C.T, (XABC - t).T)).T
    X_o_AB = np.hstack((X_o_AB, np.zeros((3, 1))))
    
    #X_o_A=np.dot(X_o_AB,B.T)
    
    #X_o=np.dot(X_o_A,A.T)
    X_o=np.dot(X_o_AB,AB.T)
    
    X_=X_o-O

    return X_

def send_mean_back(means,origin_information):
    k=5
    csv={}
    #df_final = pd.DataFrame(columns=[0,1,2,3],index=UID_names)
    final_triangles=[]
    print(means[0])
    for i in range(k):
        for x in means[i][2]:
            print(means[i][1],x)
            if means[i][1]>x:
                y=x
            else: 
                y=means[i][1]
            fin_tri=find_X(means[i][0],origin_information[y,x])
            final_triangles.append(fin_tri)
            
            
            #print(fin_tri)
            final_tri_reshape=fin_tri.reshape(1,9)
            rhino_final=['{'+str(final_tri_reshape[0][x*3])+','+str(final_tri_reshape[0][x*3+1])
                                                                            +','+str(final_tri_reshape[0][x*3+2])+'}' for x in range(3)]
           # df_final.loc[UID_names[x]]=rhino_final+[f'{i}']


FILE_NAME='TYPE_4_EXCEL_OLD.csv'


def main():
    triangle_list, orig_info = clean_csv(FILE_NAME,lenght=10)
    rotations, translations = vectorized_test(triangle_list)
    rotated_tris = rotate_triangles(rotations, translations, triangle_list)
    distance_matrix = distance_matrix_calc(rotated_tris, triangle_list)
    print("Done")


if __name__ == "__main__":
    main()
    

