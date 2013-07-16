package BO.kernel;

import org.jblas.MatrixFunctions;
import org.jblas.DoubleMatrix; 
import org.jblas.ranges.IntervalRange;


public class CovSEARD extends CovModel{
	
	public static void main(String args[]) {
		DoubleMatrix a = new DoubleMatrix(new double[] {0.1, 0.1}).transpose();
		DoubleMatrix b = new DoubleMatrix(new double[] {0.2, 0.1}).transpose();
		
		DoubleMatrix hyper = new DoubleMatrix(new double[] {0.1, 0.1, 1});
		CovSEARD csa = new CovSEARD(hyper);
		csa.cov(a, b);
	}
	
	private DoubleMatrix hyp = null;
	private DoubleMatrix ell = null;
	private double sf2 = 0;
	
	/**
	 * @param hyp is the hyper-parameters of the kernel.
	 */
	public CovSEARD(DoubleMatrix hyp) {
		this.hyp = hyp;
		IntervalRange range = new IntervalRange(0,  this.hyp.rows-1);
		this.ell = this.hyp.get(range, 0);
		this.sf2 = this.hyp.get(this.hyp.rows-1);
	}
	
	/**
	 * @param x is an n by D matrix
	 * @param y is an m by D matrix
	 * @return Kernel matrix
	 */
	public DoubleMatrix cov(DoubleMatrix x, DoubleMatrix y) {
		DoubleMatrix di = DoubleMatrix.diag(this.ell.rdiv(1));
		DoubleMatrix K = Utils.SquareDistance(di.mmul(x.transpose()), 
				di.mmul(y.transpose()));
		return MatrixFunctions.expi(K.neg().div(2.0)).mul(this.sf2);
	}
	
	/**
	 * @param is an n by D matrix
	 * @return Kernel matrix
	 */
	public DoubleMatrix cov(DoubleMatrix x) {
		return cov(x, x);
	}
	
	
	
}



