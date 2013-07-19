package hmc;

import lbfgsb.*;
import java.util.ArrayList;
import lbfgsb.Minimizer;
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
	private DoubleMatrix initPt;
	private BayesOpt bo;
	
	public int totalRandomStep = 0;
	public DoubleMatrix samples = null;
	public boolean adjustReward = true;
	
	
	public AHMC(int numIterations, int burnIn, DoubleMatrix bound, 
			MultiVariateObj gradient, Objective func, int D) {
		this.burnIn = burnIn;
		this.numIterations = numIterations;
		this.sizeAdapt = (int)Math.floor(this.burnIn/100.0);
		this.bound = bound;
		this.D = D;
		this.gradient = gradient;
		this.fun = func;
		this.initPt = this.initPoint();
		this.samples = DoubleMatrix.zeros(this.numIterations-this.burnIn, this.D);
		
		CovModel cov = new CovSEARD(hyper);
		this.bo = new BayesOpt(0.0, initPt, cov, bound, noise);
		this.bo.setUseScale(true);
		this.bo.setSooIter(200);
	}
	
	private DoubleMatrix initPoint() {
		LogDensity fun = new LogDensity(this.fun, this.gradient);
		Minimizer alg = new Minimizer();
		ArrayList<Bound> bounds = new ArrayList<Bound>();
		
		for (int i = 0; i < this.D ; i++) {
			bounds.add(new Bound(null, null));
		}
		alg.setBounds(bounds);
		
		try {
			Result result = alg.run(fun, new double[this.D]);
			return new DoubleMatrix(result.point).transpose();
			
		} catch (LBFGSBException e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	private DoubleMatrix convert(double epsilon, int L) {
		DoubleMatrix param = new DoubleMatrix(new double[]{Math.log(epsilon), L});
		DoubleMatrix ptEval = param.sub(this.bound.getColumn(0))
				.div(this.bound.getColumn(1).sub(
				this.bound.getColumn(0))).transpose();
		return ptEval;
	}
	
	public void sample() {
		this.totalRandomStep = 0;
		DoubleMatrix meanParam = this.bound.rowMeans();
		double epsilon = Math.exp(meanParam.toArray()[0]);
		int L = (int) Math.ceil(meanParam.toArray()[1]);
		int numAdapt = 0; double reward = 0;
		DoubleMatrix ptEval = convert(epsilon, L); 
		DoubleMatrix sample = this.initPt.transpose(); 
		
		for (int ii = 0; ii < this.numIterations; ii++) {
			if (ii % this.sizeAdapt == 0) {
				if (this.adjustReward) {reward = reward / Math.sqrt((double)L);}
				numAdapt = numAdapt + 1;
				if (ii >= this.sizeAdapt) {this.bo.updateModel(ptEval, reward);}
				double rate = anneal(ii);
				
				if (DoubleMatrix.rand(1).toArray()[0] < rate) {
					DoubleMatrix nextPt = this.bo.maximizeAcq(rate).
							mul(this.bound.getColumn(1).
							sub(this.bound.getColumn(0))).
							add(this.bound.getColumn(0));
					epsilon = Math.exp(nextPt.toArray()[0]);
					L = (int) Math.ceil(nextPt.toArray()[1]);
					ptEval = convert(epsilon, L);
				}
				System.out.format("Iter %3d L: %3d epsilon: %2.3f " +
						"reward: %3.2f prob: %1.3f\n", 
						numAdapt, L, epsilon, reward, rate);
				reward = 0;
			}
			DataStruct result = HMC.doIter(L, epsilon, sample, 
					this.gradient, this.fun);
			sample = result.next_q;
			reward = reward + Math.pow(result.proposal.
					sub(result.q).norm2(), 2)*result.mr;
			
			if (ii >= this.burnIn) {
				this.samples.putRow(ii-this.burnIn, sample.transpose());
				this.totalRandomStep = this.totalRandomStep + result.RandomStep;
			}
		}
	}
	
	public double anneal(int ii) {
		double anneal = Math.pow(Math.max( (ii - this.burnIn) / 
				((double) this.sizeAdapt) + 1.0, 1.0), -0.5);
		return anneal;
	}
	
	public static void main(String args[]) {
		DoubleMatrix targetSigma = new DoubleMatrix(new double[][]
				{{1.0, 0.99}, {0.99, 1.0}});
		DoubleMatrix targetMean = new DoubleMatrix( new double[] {3.0, 5.0});
		GaussianExample ge = new GaussianExample(targetSigma, targetMean);
		double[][] ba = {{Math.log(1e-3), Math.log(0.2)}, {1.0, 100.0}};
		DoubleMatrix bound = new DoubleMatrix(ba);
		
		AHMC ahmc = new AHMC(3000, 1000, bound, ge, ge, 2);
		ahmc.sample();
		ahmc.samples.columnMeans().print();
	}
}

class LogDensity implements DifferentiableFunction{
	
	private Objective func;
	private MultiVariateObj gradient;
	
	public LogDensity(Objective func, MultiVariateObj gradient) {
		this.func = func;
		this.gradient = gradient;
	}
	
	public FunctionValues getValues(double[] point){
		DoubleMatrix vec = new DoubleMatrix(point);
		double f = this.func.functionValue(vec);
		double[] grad = this.gradient.mFunctionValue(vec).toArray();
		return new FunctionValues(f, grad);
	}
}