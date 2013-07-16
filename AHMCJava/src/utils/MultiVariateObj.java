package utils;

import org.jblas.DoubleMatrix;

public interface MultiVariateObj {
	public DoubleMatrix functionValue(DoubleMatrix vec);
}
