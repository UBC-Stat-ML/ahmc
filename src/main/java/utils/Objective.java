package utils;

import org.jblas.DoubleMatrix;

public interface Objective {
	public double functionValue(DoubleMatrix vec);
}
