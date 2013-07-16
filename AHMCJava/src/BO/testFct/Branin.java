package BO.testFct;

import org.jblas.DoubleMatrix;

import utils.Objective;

public class Branin implements Objective{
	public double functionValue(DoubleMatrix vec) {
		double x = vec.get(0);
		double y = vec.get(1);
		
		double f = Math.pow(y - 5.1*Math.pow(x, 2)/(4.0* Math.pow(Math.PI, 2)) 
				+ 5.0*x/Math.PI - 6.0, 2) + 10*(1 - 1.0/(8.0*Math.PI))*
				Math.cos(x) + 10.0;
		return -f;		
	}
	
	public static void main(String args[]) {
		Branin b = new Branin();
		DoubleMatrix vec = new 
				DoubleMatrix(new double[] {-Math.PI, 12.275}).transpose();
		vec.print();
		System.out.println(b.functionValue(vec));
	}
}
