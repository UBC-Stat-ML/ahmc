package hmc;

//import lbfgsb.*;

import java.util.ArrayList;
import java.util.Random;

//import lbfgsb.Minimizer;
import org.jblas.DoubleMatrix;
import utils.MultiVariateObj;
import utils.Objective;

import bayonet.opt.LBFGSMinimizer;
import bo.BayesOpt;
import bo.kernel.CovModel;
import bo.kernel.CovSEARD;
import briefj.BriefLog;

//import lbfgsb.Bound;

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
  private double epsilon;
  private int L;
  
	public static AHMC initializeAHMCWithLBFGS(int numIterations, int burnIn, MultiVariateObj gradient, 
      Objective func, int D) {
	  return new AHMC(numIterations, burnIn, defaultEpsilonLBounds(), gradient, func, null, D);
	}
	
	public AHMC(int numIterations, int burnIn, MultiVariateObj gradient, 
	    Objective func, double [] initialPoint) {
	  this(numIterations, burnIn, defaultEpsilonLBounds(), gradient, func, initialPoint, initialPoint.length);
	}
	
	public AHMC(int numIterations, int burnIn, DoubleMatrix bound, 
			MultiVariateObj gradient, Objective func, double [] initialPoint, int D) {
		this.burnIn = burnIn;
		this.numIterations = numIterations;
		this.sizeAdapt = (int)Math.floor(this.burnIn/100.0);
		this.bound = bound;
		this.D = D;
		this.gradient = gradient;
		this.fun = func;
		if (initialPoint == null)
		  initialPoint = initPoint_pureJava();
	  this.initPt = new DoubleMatrix(initialPoint).transpose();	
		CovModel cov = new CovSEARD(hyper);
		
		this.bo = new BayesOpt(0.0, convert(initEpsilon(), initL()), cov, bound, noise);
		this.bo.setUseScale(true);
		this.bo.setSooIter(200);
	}
	


  /**
	 * Call this before sample() to request all samples
	 * be kept in a matrix.
	 */
	public void keepSamples()
	{
	  this.samples = DoubleMatrix.zeros(this.numIterations-this.burnIn, this.D);
	}
	
	private double [] initPoint_pureJava() {
	  
	  LBFGSMinimizer minimizer = new LBFGSMinimizer();
	  
	  bayonet.opt.DifferentiableFunction f = new bayonet.opt.DifferentiableFunction() {

      @Override
      public int dimension()
      {
        return AHMC.this.D;
      }

      @Override
      public double valueAt(double[] x)
      {
        return AHMC.this.fun.functionValue(new DoubleMatrix(x));
      }

      @Override
      public double[] derivativeAt(double[] x)
      {
        return AHMC.this.gradient.mFunctionValue(new DoubleMatrix(x)).data;
      }
	    
	  };
	  
	  return minimizer.minimize(f, new double[this.D], 1e-5);
	}
	
	private DoubleMatrix convert(double epsilon, int L) {
		DoubleMatrix param = new DoubleMatrix(new double[]{Math.log(epsilon), L});
		DoubleMatrix ptEval = param.sub(this.bound.getColumn(0))
				.div(this.bound.getColumn(1).sub(
				this.bound.getColumn(0))).transpose();
		return ptEval;
	}
	
	private double initEpsilon() {
    return Math.exp(this.bound.rowMeans().toArray()[0]);
  }
	
	private int initL() {
	  return (int) Math.ceil(this.bound.rowMeans().toArray()[1]);
	}
	
	public DoubleMatrix sample(Random rand) {
		this.totalRandomStep = 0;
		epsilon = initEpsilon();
		L = initL();
		int numAdapt = 0; double reward = 0;
		DoubleMatrix ptEval = convert(epsilon, L); 
		DoubleMatrix sample = this.initPt.transpose(); 
		
		for (int ii = 0; ii < this.numIterations; ii++) {
			if (ii % this.sizeAdapt == 0) {
				if (this.adjustReward) {reward = reward / Math.sqrt((double)L);}
				numAdapt = numAdapt + 1;
				if (ii >= this.sizeAdapt) {this.bo.updateModel(ptEval, reward);}
				double rate = anneal(ii);
				
				if (rand.nextDouble() < rate) {
					DoubleMatrix nextPt = this.bo.maximizeAcq(rate).
							mul(this.bound.getColumn(1).
							sub(this.bound.getColumn(0))).
							add(this.bound.getColumn(0));
					epsilon = Math.exp(nextPt.toArray()[0]);
					L = (int) Math.ceil(nextPt.toArray()[1]);
					ptEval = convert(epsilon, L);
				}
				System.out.format("Iter %3d L: %3d epsilon: %f " +
						"reward: %f prob: %f\n", 
						numAdapt, L, epsilon, reward, rate);
				reward = 0;
			}
			DataStruct result = HMC.doIter(rand, L, epsilon, sample, 
					this.gradient, this.fun);
			sample = result.next_q;
			reward = reward + (result.mr == 0.0 ? 0.0 : Math.pow(result.proposal.
					sub(result.q).norm2(), 2)*result.mr);
			
			if (ii >= this.burnIn) {
			  if (this.samples != null)
			    this.samples.putRow(ii-this.burnIn, sample.transpose());
				this.totalRandomStep = this.totalRandomStep + result.RandomStep;
			}
		}
		return sample;
	}
	
	public double anneal(int ii) {
		double anneal = Math.pow(Math.max( (ii - this.burnIn) / 
				((double) this.sizeAdapt) + 1.0, 1.0), -0.5);
		return anneal;
	}
	
	public static DoubleMatrix defaultEpsilonLBounds()
	{
	  double[][] ba = {{Math.log(1e-3), Math.log(0.2)}, {1.0, 100.0}};
    DoubleMatrix bound = new DoubleMatrix(ba);
    return bound;
	}
	
	public static void main(String args[]) {
	  Random rand = new Random(1);
		DoubleMatrix targetSigma = new DoubleMatrix(new double[][]
				{{1.0, 0.99}, {0.99, 1.0}});
		DoubleMatrix targetMean = new DoubleMatrix( new double[] {3.0, 5.0});
		GaussianExample ge = new GaussianExample(targetSigma, targetMean);
		
		
//		AHMC ahmc = new AHMC(3000, 1000, bound, ge, ge, 2);
		
		AHMC ahmc = initializeAHMCWithLBFGS(3000, 1000, ge, ge, 2);
		
		ahmc.keepSamples();
		ahmc.sample(rand);
		ahmc.samples.columnMeans().print();
		
		ahmc.sample(rand);
		ahmc.samples.columnMeans().print();
	}

  public double getEpsilon()
  {
    return epsilon;
  }

  public int getL()
  {
    return L;
  }
}

//class LogDensity {//implements DifferentiableFunction{
//	
//	private Objective func;
//	private MultiVariateObj gradient;
//	
//	public LogDensity(Objective func, MultiVariateObj gradient) {
//		this.func = func;
//		this.gradient = gradient;
//	}
//	
//	public FunctionValues getValues(double[] point){
//		DoubleMatrix vec = new DoubleMatrix(point);
//		double f = this.func.functionValue(vec);
//		double[] grad = this.gradient.mFunctionValue(vec).toArray();
//		return new FunctionValues(f, grad);
//	}
//}