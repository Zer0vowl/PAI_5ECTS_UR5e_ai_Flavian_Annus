# %%
def evaluate_trajectory(Y_ref, Y_dmp):
    """
    Returns a dict with:
      - RMS positional error
      - max positional error
      - per-axis Pearson correlation
      - covariance trace (just as an extra similarity hint)
    """
    assert Y_ref.shape == Y_dmp.shape
    diff = Y_dmp - Y_ref
    # Euclidean error per time step
    err = np.linalg.norm(diff, axis=1)

    rms_err = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(err))

    # Per-axis correlations (if variance > 0)
    corr = []
    for i in range(Y_ref.shape[1]):
        r = np.corrcoef(Y_ref[:, i], Y_dmp[:, i])[0, 1]
        corr.append(float(r))

    # Covariance between flattened position vectors (just a rough scalar)
    X = np.vstack([Y_ref.ravel(), Y_dmp.ravel()])
    cov_mat = np.cov(X)
    cov_trace = float(np.trace(cov_mat))

    return {
        "rms_error": rms_err,
        "max_error": max_err,
        "corr_x": corr[0],
        "corr_y": corr[1],
        "corr_z": corr[2],
        "cov_trace": cov_trace,
    }
