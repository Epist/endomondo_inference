def median_smoothing(seq, context_size):
	seq_len = len(seq)

	if context_size%2==0:
		raise(exception("Context size must be odd for median smoothing"))

	smoothed_seq = []
	for i in range(seq_len):
		cont_diff = (context_size-1)/2
		context_min = max(0, i-cont_diff)
		context_max = min(seq_len, i+cont_diff)
		median_val = np.median(seq[context_min:context_max])
		smoothed_seq.append(median_val)

	return smoothed_seq
