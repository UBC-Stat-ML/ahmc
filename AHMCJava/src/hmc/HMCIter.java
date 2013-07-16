package hmc;

import org.jblas.DoubleMatrix;
import utils.MultiVariateObj;
import utils.Objective;

public class HMCIter {
	public static DataStruct doIter(int l, double epsilon, DoubleMatrix lastSample, 
			MultiVariateObj gradient, Objective func){
		int D = lastSample.rows;
		int randomStep = (int)Math.ceil(DoubleMatrix.rand(1).toArray()[0]*l);
		
		DoubleMatrix old_p = DoubleMatrix.rand(D).transpose(); // Momentum Vec
		DoubleMatrix p = old_p.sub(gradient.functionValue(lastSample)
				.mmul(epsilon*0.5));
		
		DoubleMatrix q = lastSample;
		
		for (int ii = 0; ii < randomStep; ii++) {
			q = q.add(p.mmul(epsilon));
			p = p.sub(gradient.functionValue(q)
					.mmul(epsilon));
		}
		p = p.add(gradient.functionValue(q)
				.mmul(epsilon*0.5));
		
		double proposed_E = func.functionValue(q); 
		double original_E = func.functionValue(lastSample);
		
		double proposed_K = p.transpose().mmul(p).div(2.0).toArray()[0]; 
		double original_K = old_p.transpose().mmul(old_p).div(2.0).toArray()[0];
		
		double mr = -proposed_E +  original_E + original_K - proposed_K;
		
		if (Double.isNaN(mr)) {
			mr = Math.min(mr, 1.0);
		} else {
			mr = 0.0;
		}
		
		boolean accept = true;
		double energy = -proposed_E;
		if (DoubleMatrix.rand(1).toArray()[0] > mr) {
			q = lastSample;
			accept = false;
			energy = -original_E;
		}
		
		return new DataStruct(q, accept, lastSample, mr, randomStep, energy);
	}
	
	
}
