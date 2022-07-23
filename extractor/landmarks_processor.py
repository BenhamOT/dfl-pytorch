import cv2
import numpy as np


def get_transform_mat(image_landmarks, output_size, scale=1.0):
    # estimate landmarks transform from global space to local aligned space with bounds [0..1]
    mat = umeyama(np.concatenate([image_landmarks[17:49], image_landmarks[54:55]]), landmarks_2D_new, True)[0:2]

    # get corner points in global space
    g_p = transform_points(np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]), mat, True)
    g_c = g_p[4]

    # calc diagonal vectors between corners in global space
    tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
    tb_diag_vec /= np.linalg.norm(tb_diag_vec)
    bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
    bt_diag_vec /= np.linalg.norm(bt_diag_vec)

    # calc modifier of diagonal vectors for scale and padding value
    padding = 0.2109375
    mod = (1.0 / scale) * (np.linalg.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))

    # calc 3 points in global space to estimate 2d affine transform
    l_t = np.array([g_c - tb_diag_vec * mod,
                    g_c + bt_diag_vec * mod,
                    g_c + tb_diag_vec * mod])

    # calc affine transform from 3 global space points to 3 local space points size of 'output_size'
    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    mat = cv2.getAffineTransform(l_t, pts2)
    return mat


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def umeyama(src, dst, estimate_scale):
    """
    Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array(lmrks.copy(), dtype=np.int)

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_image_hull_mask(image_shape, image_landmarks, eyebrows_expand_mod=1.0):
    hull_mask = np.zeros(image_shape[0:2] + (1,), dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks, eyebrows_expand_mod)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), (1,))

    return hull_mask


def get_image_eye_mask(image_shape, image_landmarks):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    h, w, c = image_shape
    hull_mask = np.zeros((h, w, 1), dtype=np.float32)
    image_landmarks = image_landmarks.astype(np.int)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(image_landmarks[36:42]), (1,))
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(image_landmarks[42:48]), (1,))

    dilate = h // 32
    hull_mask = cv2.dilate(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)), iterations=1)

    blur = h // 16
    blur = blur + (1 - blur % 2)
    hull_mask = cv2.GaussianBlur(hull_mask, (blur, blur), 0)
    hull_mask = hull_mask[..., None]
    return hull_mask


landmarks_2D_new = np.array([
    [0.000213256, 0.106454],  # 17
    [0.0752622, 0.038915],  # 18
    [0.18113, 0.0187482],  # 19
    [0.29077, 0.0344891],  # 20
    [0.393397, 0.0773906],  # 21
    [0.586856, 0.0773906],  # 22
    [0.689483, 0.0344891],  # 23
    [0.799124, 0.0187482],  # 24
    [0.904991, 0.038915],  # 25
    [0.98004, 0.106454],  # 26
    [0.490127, 0.203352],  # 27
    [0.490127, 0.307009],  # 28
    [0.490127, 0.409805],  # 29
    [0.490127, 0.515625],  # 30
    [0.36688, 0.587326],  # 31
    [0.426036, 0.609345],  # 32
    [0.490127, 0.628106],  # 33
    [0.554217, 0.609345],  # 34
    [0.613373, 0.587326],  # 35
    [0.121737, 0.216423],  # 36
    [0.187122, 0.178758],  # 37
    [0.265825, 0.179852],  # 38
    [0.334606, 0.231733],  # 39
    [0.260918, 0.245099],  # 40
    [0.182743, 0.244077],  # 41
    [0.645647, 0.231733],  # 42
    [0.714428, 0.179852],  # 43
    [0.793132, 0.178758],  # 44
    [0.858516, 0.216423],  # 45
    [0.79751, 0.244077],  # 46
    [0.719335, 0.245099],  # 47
    [0.254149, 0.780233],  # 48
    [0.726104, 0.780233],  # 54
], dtype=np.float32)
