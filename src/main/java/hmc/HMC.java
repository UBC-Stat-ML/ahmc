package hmc;

import java.util.Random;

import org.jblas.DoubleMatrix;
import utils.MultiVariateObj;
import utils.Objective;

import org.jblas.Solve;

public class HMC {
	
	private int l;
	private double epsilon;
	private MultiVariateObj gradient;
	private Objective func;
	
	public HMC(int l, double epsilon, MultiVariateObj gradient, 
			Objective func) {
		this.l = l;
		this.epsilon = epsilon;
		this.gradient = gradient;
		this.func = func;
	}
	
	public DoubleMatrix run(Random rand, int burnIn, int totalNumSample, DoubleMatrix sample) {
		
		DoubleMatrix samples = DoubleMatrix.zeros(totalNumSample-burnIn, 
				sample.rows);
		
		for (int i = 0 ; i < totalNumSample; i++) {
			DataStruct result = doIter(rand, l, epsilon, sample, gradient, func);
			sample = result.next_q;
			
			if ((i+1)%100 == 0) {
				System.out.println("Iteration " + (i+1) + ".");
			}
			
			if (i >= burnIn) {
				samples.putRow(i-burnIn, sample.transpose());
			}
		}
		
		return samples;
	}
	
	public static DataStruct doIter(Random rand, int l, double epsilon, 
      DoubleMatrix lastSample, MultiVariateObj gradient, Objective func)
  {
	  return doIter(rand, l, epsilon, lastSample, gradient, func, true);
  }
	
	public static DataStruct doIter(Random rand, int l, double epsilon, 
			DoubleMatrix lastSample, MultiVariateObj gradient, Objective func,boolean randomizeNumberOfSteps){
		
		int D = lastSample.rows;
		int nSteps = randomizeNumberOfSteps ? (int)Math.ceil(rand.nextDouble()*l) : l;
		
		DoubleMatrix proposal = lastSample;

		// Generate Momentum Vector
		DoubleMatrix old_p = new DoubleMatrix(D,1); 
		for (int i = 0; i < D; i++)
		  old_p.put(i, 0, rand.nextGaussian());
		
		DoubleMatrix p = old_p.sub(gradient.mFunctionValue(proposal)
				.mmul(epsilon*0.5));
		for (int ii = 0; ii < nSteps; ii++) {
			proposal = proposal.add(p.mmul(epsilon));
			p = p.sub(gradient.mFunctionValue(proposal)
					.mmul(epsilon));
		}
		p = p.add(gradient.mFunctionValue(proposal)
				.mmul(epsilon*0.5));
		
		double proposed_E = func.functionValue(proposal); 
		double original_E = func.functionValue(lastSample);
		double proposed_K = p.transpose().mmul(p).div(2.0).toArray()[0]; 
		double original_K = old_p.transpose().mmul(old_p).div(2.0).toArray()[0];
		double mr = -proposed_E +  original_E + original_K - proposed_K;
		
		if (!Double.isNaN(mr)) {
			mr = Math.min(Math.exp(mr), 1.0);
		} else {
			mr = 0.0;
		}
		
		DoubleMatrix nextSample = proposal;
		
		boolean accept = true;
		double energy = -proposed_E;
		if (rand.nextDouble() > mr) {
			nextSample = lastSample;
			accept = false;
			energy = -original_E;
		}
		
		return new DataStruct(nextSample, accept, proposal, lastSample, 
				mr, nSteps, energy);
	}
	
	public static void main(String[] args) {
	  Random rand = new Random(1);
		DoubleMatrix targetSigma = new DoubleMatrix(new double[][]
				{{1.0, 0.99}, {0.99, 1.0}});
		DoubleMatrix targetMean = new DoubleMatrix( new double[]{3.0, 5.0});
		GaussianExample ge = new GaussianExample(targetSigma, targetMean);
		HMC hmc = new HMC(40, 0.05, ge, ge);
		DoubleMatrix sample = new DoubleMatrix( new double[]{3.0, 5.0});
		DoubleMatrix samples = hmc.run(rand, 0, 2000, sample);
		samples.columnMeans().print();
	}
}

class GaussianExample implements MultiVariateObj, Objective {
	private DoubleMatrix targetSigma;
	private DoubleMatrix targetMean;
	
	public GaussianExample(DoubleMatrix targetSigma, DoubleMatrix targetMean) {
		this.targetSigma = targetSigma;
		this.targetMean = targetMean;
	}
	
	public double functionValue(DoubleMatrix vec) {
		DoubleMatrix diff = vec.sub(this.targetMean);
		return diff.transpose().mmul(Solve.solve(this.targetSigma, diff)).
				mmul(0.5).toArray()[0];
	}
	
	public DoubleMatrix mFunctionValue(DoubleMatrix vec) {
		DoubleMatrix diff = vec.sub(this.targetMean);
		return Solve.solve(this.targetSigma, diff);
	}
}
