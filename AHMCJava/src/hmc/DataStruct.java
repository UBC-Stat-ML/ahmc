package hmc;

import org.jblas.DoubleMatrix;

public class DataStruct {
	
	public DoubleMatrix next_q = null;
	public boolean accept = false;
	public DoubleMatrix q = null; 
	public double mr = 0;
	public int RandomStep = -1; 
	public double energy = 0;
	
	public DataStruct(DoubleMatrix next_q, boolean accept, DoubleMatrix q, 
			double mr, int RandomStep, double energy) {
		this.next_q = next_q;
		this.accept = accept;
		this.q = q;
		this.mr = mr;
		this.RandomStep = RandomStep;
		this.energy = energy;
	}
}
