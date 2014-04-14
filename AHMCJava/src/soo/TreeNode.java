package soo;

import java.util.ArrayList;

public class TreeNode {

	protected int depth = 0;
	protected int dim = 0;
	protected double val = -1;
	protected TreeNode parent = null;
	protected ArrayList<TreeNode> children = null;
	protected Bounds bound = null;
	protected double[] peval = null;

	public TreeNode(int dim) {
		this.dim = dim;
	}

	public TreeNode(int dim, int depth) {
		this(dim);
		this.depth = depth;
	}

	public TreeNode(int dim, int depth, double[][] bound, TreeNode parent) {
		this(dim, depth);
		this.parent = parent;
		this.bound = new Bounds(dim, bound);
	}

	public void produceChildren(int splitDim) {
		double[][] left = this.bound.copy();
		double[][] right = this.bound.copy();

		double mid = (left[splitDim][0]+left[splitDim][1])/2;
		left[splitDim][1] = mid; right[splitDim][0] = mid;

		TreeNode leftChild = new TreeNode(this.dim, this.depth+1, left, this);
		TreeNode rightChild = new TreeNode(this.dim, this.depth+1, right, this);
		this.children = new ArrayList<TreeNode>();
		this.children.add(leftChild);
		this.children.add(rightChild);
	}

	public double[] pointOfEvalution() {
		this.peval = this.bound.midpoint();
		return this.peval;
	}

	public void setEvalutionValue(double val) {
		this.val = val;
	}

	public double getEvalutionValue() {
		return this.val;
	}

	public void setBound(Bounds bound) {
		this.bound = bound;
	}

	public void initRootBound() {
		this.bound = new Bounds(this.dim);
	}

	public double getValue() {
		return this.val;
	}

	public ArrayList<TreeNode> getChildren() {
		return children;
	}

	public int getDepth() {
		return this.depth;
	}

	public double[][] getBounds() {
		return this.bound.getBounds();
	}
}


class Bounds {
	double[][] bounds = null;
	int dim = 0;

	public Bounds(int dim) {
		this.dim = dim;
		this.initialize();
	}

	public Bounds(int dim, double[][] bounds) {
		this.dim = dim;
		this.initialize(bounds);
	}

	public double[][] copy() {
		double[][] newBound = new double[this.dim][2];
		for (int i = 0; i < this.dim; i++) {
			for (int j = 0; j < 2; j++) {
				newBound[i][j] = this.bounds[i][j];
			}
		}
		return newBound;
	}

	public double[][] getBounds() {
		return this.bounds;
	}

	public double[] midpoint() {
		double[] midpoint = new double[this.dim];
		for (int i = 0; i < this.dim; i++) {
			midpoint[i] = (this.bounds[i][0] + this.bounds[i][1])/2;
		}

		return midpoint;
	}

	private void initialize() {
		// Initilize standard bounds [0, 1]^d.
		this.bounds = new double[this.dim][2];
		for (int i = 0; i < this.dim; i++) {
			for (int j = 0; j < 2; j++) {
				if (j == 0) {
					this.bounds[i][j] = 0;
				} else {
					this.bounds[i][j] = 1;
				}
				
			}
		}
	}

	private void initialize(double[][] initBounds) {
		this.bounds = initBounds;
	}
}