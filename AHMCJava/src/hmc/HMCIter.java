package hmc;

import org.jblas.DoubleMatrix;
import utils.MultiVariateObj;;

public class HMCIter {
	public static void doIter(int l, double epsilon, DoubleMatrix lastSample, 
			MultiVariateObj gradient, MultiVariateObj func){
		int D = lastSample.rows;
		int randomStep = (int)Math.ceil(DoubleMatrix.rand(1).toArray()[0]*l);
		
		DoubleMatrix old_p = DoubleMatrix.rand(D).transpose(); // Momentum Vec
		DoubleMatrix p = old_p.sub(gradient.functionValue(lastSample)
				.mmul(epsilon*0.5));
		
		DoubleMatrix q = lastSample;
		
		for (int ii = 0; ii < randomStep; ii++) {
			q = q.add(p.mmul(epsilon));
			p = old_p.sub(gradient.functionValue(q)
					.mmul(epsilon*0.5));
		}
		
		
	}
	
	
}
