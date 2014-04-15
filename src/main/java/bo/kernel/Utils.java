package bo.kernel;

import org.jblas.DoubleMatrix;

public class Utils {
	
	public static void main(String args[]) {
		DoubleMatrix a = new DoubleMatrix(new double[] {0.1, 0.1}).transpose();
		DoubleMatrix b = new DoubleMatrix(new double[] {0.2, 0.1}).transpose();
		DoubleMatrix C = Utils.SquareDistance(a, b);
		C.print();
	}
	
	/**
	 * @param a is a D by n DoubleMatrix 
	 * @param b is a D by m DoubleMatrix
	 * @return a Double Matrix that is n by m of all pairwise squared distances 
	 * between a and b.
	 */
	public static DoubleMatrix SquareDistance(DoubleMatrix a, DoubleMatrix b) {
		
		double n = (double)a.columns; double m = (double)b.columns;
		
		DoubleMatrix mu = b.rowMeans().mmul((m/(n+m))).
				add(a.rowMeans().mmul(n/(n+m)));
		a = a.sub(mu.repmat(1,(int)n));  b = b.sub(mu.repmat(1,(int)m));
		
		DoubleMatrix C = (a.mul(a).columnSums().transpose()).repmat(1,(int)m);
		C = C.add((b.mul(b).columnSums()).repmat((int)n,1));
		C = C.sub(a.transpose().mmul(b).mul(2));
		
		return C.max(0);
	}
	
	/**
	 * @param a is a D by n DoubleMatrix
	 * @return a Double Matrix that is n by n of all pairwise squared distances 
	 * between a and a.
	 */
	public static DoubleMatrix SquareDistance(DoubleMatrix a) {
		DoubleMatrix b = a;
		return SquareDistance(a, b);
	}
}

