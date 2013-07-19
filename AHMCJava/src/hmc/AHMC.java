package hmc;

import org.jblas.DoubleMatrix;
import utils.MultiVariateObj;
import utils.Objective;

import bo.BayesOpt;
import bo.kernel.CovModel;
import bo.kernel.CovSEARD;

import lbfgsb.Bound;

public class AHMC {
	private int burnIn = 0;
	private int numIterations = 0;
	private MultiVariateObj gradient = null;
	private Objective fun = null;
	private int D;
	private int sizeAdapt = 10;
	private DoubleMatrix bound;
	private double noise = 0.1;
	private DoubleMatrix hyper = 
			new DoubleMatrix(new double[] {0.2, 0.2, 5});
	
	private BayesOpt bo;
	
	public boolean adjustReward = true;
	
	
	public AHMC(int numIterations, int burnIn, DoubleMatrix bound, 
			MultiVariateObj gradient, Objective func, int D) {
		this.burnIn = burnIn;
		this.numIterations = numIterations;
		this.sizeAdapt = (int)Math.floor(this.burnIn/100.0);
		this.bound = bound;
		
		
		
		CovModel cov = new CovSEARD(hyper);
		this.bo = new BayesOpt(0.0, initPt, cov, bound, noise);
	}
	
	public void sample() {
		
		DoubleMatrix meanParam = this.bound.rowMeans();
		double epsilon = Math.exp(meanParam.toArray()[0]);
		int L = (int) Math.ceil(meanParam.toArray()[1]);
		int numAdapt = 0;
		double reward = 0;
		
		
		
		for (int ii = 0; ii < this.numIterations; ii++) {
			if (ii == 0) {
				
			} else if (ii % this.sizeAdapt == 0) {
				if (this.adjustReward) {
					reward = reward / Math.sqrt((double)L);
				}
				numAdapt = numAdapt + 1;
				
				if (ii == this.sizeAdapt) {
					
				} else {
					this.bo.updateModel(x, reward);
				}
				
				
			}
		}
	}
	
	public static void main(String[] args) {
		DoubleMatrix targetSigma = new DoubleMatrix(new double[][]
				{{1.0, 0.99}, {0.99, 1.0}});
		DoubleMatrix targetMean = new DoubleMatrix( new double[]{3.0, 5.0});
		GaussianExample ge = new GaussianExample(targetSigma, targetMean);
		double[][] ba = {{Math.log(1e-5), Math.log(1e-1)}, {1.0, 100.0}};
		DoubleMatrix bound = new DoubleMatrix(ba);
		
		AHMC ahmc = new AHMC(6000, 1000, bound, ge, ge, 2);
		ahmc.sample();
	}
}
