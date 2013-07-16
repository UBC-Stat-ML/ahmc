package BO;

import org.jblas.DoubleMatrix; 
import org.jblas.Decompose;
import org.jblas.MatrixFunctions;
import org.jblas.Solve;
import org.jblas.ranges.IntervalRange;

import soo.SOO;
import soo.TreeNode;

public class BayesOpt { 
	private double noise = 0;
	private DoubleMatrix bound = null;
	private double scale = 1;
	private int n = 1;
	private double maxVal = 0;
	private Objective obj = null;
	private DoubleMatrix X = null;
	private DoubleMatrix Y = null;
	private int D = -1;
	
	private DoubleMatrix kernelCholL = null;
	
	private CovModel covModel = null;
	
	public BayesOpt(Objective obj, DoubleMatrix initPt, CovModel covModel, 
			DoubleMatrix bound, double noise) {
		this.covModel = covModel;
		this.noise = noise;
		this.bound = bound;
		this.obj = obj;
		
		double initVal = this.obj.functionValue(initPt);
		this.D = initPt.columns;
		
		this.X = DoubleMatrix.zeros(2000, initPt.columns);
		this.Y = DoubleMatrix.zeros(2000, 1);
		X.putRow(0, initPt);
		Y.put(0, initVal);
		
		this.kernelCholL = Decompose.cholesky(this.covModel.cov(initPt, initPt).
				add(this.noise));
	}
	
	public void updateModel(DoubleMatrix x) {
		DoubleMatrix ptEval = x.mul(this.bound.getColumn(1).sub(
				this.bound.getColumn(0))).add(this.bound.getColumn(0));		
		double f = this.obj.functionValue(ptEval);
		System.out.println(f);
		this.updateKenel(x);
		this.n = this.n + 1;
		this.X.putRow(this.n-1, x);
		this.Y.put(this.n-1, f);
		
		if (f > this.maxVal) {
			this.maxVal = f;
			//this.scale = 4.0/this.maxVal;
		}
	}
	
	
	public void updateKenel(DoubleMatrix x) {
		IntervalRange rangeR = new IntervalRange(0,  this.n);
		IntervalRange rangeC = new IntervalRange(0,  this.n);
		IntervalRange endRange = new IntervalRange(this.n,  this.n+1);
		DoubleMatrix k_x = this.covModel.cov(this.X.getRows(rangeR), x);
		DoubleMatrix k_tt = this.covModel.cov(x, x);
		
		
		DoubleMatrix z_t = Solve.solve(this.kernelCholL, k_x);
		DoubleMatrix d_t = MatrixFunctions.sqrt(k_tt.add(this.noise).
				sub(z_t.transpose().mmul(z_t)));
		
		DoubleMatrix newKernelCholL = DoubleMatrix.zeros(this.n+1, this.n+1);
		newKernelCholL.put(rangeR, rangeC, this.kernelCholL);
		newKernelCholL.put(endRange, rangeR, z_t.transpose());
		newKernelCholL.put(endRange, endRange, d_t);
		this.kernelCholL = newKernelCholL;
	}
	
	public Pair<DoubleMatrix, DoubleMatrix> meanVar(DoubleMatrix x) {
		IntervalRange range = new IntervalRange(0,  this.n);
		DoubleMatrix k_x = this.covModel.cov(this.X.getRows(range), x);
		DoubleMatrix k_tt = this.covModel.cov(x, x);
		DoubleMatrix intermediate = Solve.solve(this.kernelCholL.transpose(),
				Solve.solve(this.kernelCholL, k_x)).transpose();
		DoubleMatrix mean = intermediate.
				mmul(this.Y.get(range, 0)).mmul(this.scale);
		DoubleMatrix var = k_tt.sub(intermediate.mmul(k_x));
		
		return new Pair<DoubleMatrix, DoubleMatrix>(mean ,var);
		
	}
	
	public DoubleMatrix ucb(DoubleMatrix x) {
		return this.ucb(x, 1.0);
	}
	
	public DoubleMatrix ucb(DoubleMatrix x, double si) {
		Pair<DoubleMatrix, DoubleMatrix> mv = this.meanVar(x);
		DoubleMatrix std = MatrixFunctions.sqrt(mv.getRight());
		double ndouble = (double)this.n;
		double d = (double)this.D;
		double coeff = si*Math.sqrt(2.0*Math.log( Math.pow(ndouble, (d/2.0+2.0))* 
				Math.pow(Math.PI, 2.0) / (3.0*0.1)) );
		
		DoubleMatrix acq = (mv.getLeft().add(std.mmul(coeff)));
		return acq;
	}
	
	public DoubleMatrix maximizeAcq() {
		Acquisition acq = new Acquisition(this);
		SOO opt = new SOO(acq, this.D);
		
		double maxVal = Double.NEGATIVE_INFINITY;
		TreeNode bestNode = null;
		for (int ii = 0; ii < 500; ii++) {
			TreeNode node = opt.next();
			double val = node.getEvalutionValue();
			if (val > maxVal) {
				maxVal = val;
				bestNode = node;
			}
		}
		return (new DoubleMatrix(bestNode.pointOfEvalution())).transpose();
	}
	
	public void maximize(int maxIter) {
		for (int i = 0; i < maxIter; i++) {
			System.out.println(i);
			DoubleMatrix pt = this.maximizeAcq();
			pt.print();
			updateModel(pt);
		}
		IntervalRange range = new IntervalRange(0,  this.n);
		System.out.println(this.Y.get(range, 0).max());
	}
	
	public static void main(String args[]) {
		DoubleMatrix initPt = DoubleMatrix.rand(2).transpose();
		DoubleMatrix hyper = new DoubleMatrix(new double[] {0.072, 0.072, 1});
		CovModel cov = new CovSEARD(hyper);
		double noise = 1e-6;
		double[][] ba = {{-5.0, 10.0}, {0.0, 15.0}};
		DoubleMatrix bound = new DoubleMatrix(ba);
		Objective branin = new Branin();
		
		BayesOpt bo = new BayesOpt(branin, initPt, cov, bound, noise);
		
		bo.maximize(100);
		System.out.println("Finished");
	}
}

class Acquisition implements Objective {

	private BayesOpt bo = null;
	
	public Acquisition(BayesOpt bo) {
		this.bo = bo;
	}
	
	@Override
	public double functionValue(DoubleMatrix vec) {
		return this.bo.ucb(vec.transpose()).toArray()[0];
	}
	
}

