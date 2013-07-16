package hmc;

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
	
	public DoubleMatrix run(int burnIn, int totalNumSample, DoubleMatrix sample) {
		
		DoubleMatrix samples = DoubleMatrix.zeros(totalNumSample-burnIn, 
				sample.rows);
		
		for (int i = 0 ; i < totalNumSample; i++) {
			DataStruct result = doIter(l, epsilon, sample, gradient, func);
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
	
	public static DataStruct doIter(int l, double epsilon, 
			DoubleMatrix lastSample, MultiVariateObj gradient, Objective func){
		
		int D = lastSample.rows;
		int randomStep = (int)Math.ceil(DoubleMatrix.rand(1).toArray()[0]*l);
		
		DoubleMatrix proposal = lastSample;

		// Generate Momentum Vector
		DoubleMatrix old_p = DoubleMatrix.randn(D).transpose();
		DoubleMatrix p = old_p.sub(gradient.mFunctionValue(proposal)
				.mmul(epsilon*0.5));
		for (int ii = 0; ii < randomStep; ii++) {
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
		
		boolean accept = true;
		double energy = -proposed_E;
		if (DoubleMatrix.rand(1).toArray()[0] > mr) {
			proposal = lastSample;
			accept = false;
			energy = -original_E;
		}
		
		return new DataStruct(proposal, accept, lastSample, mr, randomStep, 
				energy);
	}
	
	public static void main(String[] args) {
		DoubleMatrix targetSigma = new DoubleMatrix(new double[][]
				{{1.0, 0.99}, {0.99, 1.0}});
		DoubleMatrix targetMean = new DoubleMatrix( new double[]{3.0, 5.0});
		GaussianExample ge = new GaussianExample(targetSigma, targetMean);
		HMC hmc = new HMC(40, 0.05, ge, ge);
		DoubleMatrix sample = new DoubleMatrix( new double[]{3.0, 5.0});
		hmc.run(0, 1000, sample);
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
