package soo;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Iterator;

import BO.Objective;
import org.jblas.DoubleMatrix;

public class SOO {

	protected Levels levels;
	protected TreeNode rootNode;
	protected int dim = 0;
	protected int expansionCount = 0;
	protected Queue<TreeNode> evalutionQueue = new LinkedList<TreeNode>();
	protected int curDepth = 0;
	protected double maxLoopingVal = Double.NEGATIVE_INFINITY;
	protected Objective objFct = null;
	
	public static void main(String args[]) {
		SOO opt = new SOO(1);
		for (int kk = 0; kk < 30; kk++) {
			double a = opt.next().getEvalutionValue();
			System.out.println(a);
		}
	}

	public SOO(int dim) {
		this.dim = dim;							// Dimension of the opt problem
		levels = new Levels();
		this.initialize();
	}
	
	public SOO(Objective obj, int dim) {
		this.objFct = obj;
		this.dim = dim;							// Dimension of the opt problem
		levels = new Levels();
		this.initialize();
	}

	protected void initialize() {
		this.rootNode = new TreeNode(dim);		// Rootnode
		this.rootNode.initRootBound();
		this.evalutionQueue.offer(this.rootNode);
	}

	public int getHMax() {
		return (int)Math.ceil(Math.sqrt((double)this.expansionCount))+1;
	}

	public double evaluate(double[] point) {
		if (this.objFct != null) {
			return this.objFct.functionValue(new DoubleMatrix(point));
		}else {
			return Math.sin(4*point[0]);
		}

	}
	
	public void incrementDepth() {
        this.curDepth = (this.curDepth + 1)%Math.min(this.levels.getNumLevels(), 
            this.getHMax());
	}

	public int getExpansionCount() {
		return this.expansionCount;
	}

	public TreeNode next() {
		while (this.evalutionQueue.peek() == null) {
			
			if (this.curDepth == 0) {
				this.maxLoopingVal = Double.NEGATIVE_INFINITY;
			}

			Level curLevel = this.levels.getLevel(curDepth);
			double curLevelMaxVal = curLevel.getMaxVal();
			

			// If level does not have a node then skip.
			if (Double.isInfinite(curLevelMaxVal) && curLevelMaxVal < 0) {
				// Go down a level.
				this.curDepth = (this.curDepth + 1)%Math.min(this.levels.getNumLevels(), 
					this.getHMax());
				continue;
			}

			// Sample point.
			if (this.maxLoopingVal <= curLevelMaxVal) {
				TreeNode toExpand = curLevel.getMaxNode();
				int splitDim = curDepth%this.dim;
				toExpand.produceChildren(splitDim);

				// Add Children to Queue.
				Iterator<TreeNode> itr = toExpand.getChildren().iterator();
				while (itr.hasNext()) {
					this.evalutionQueue.offer(itr.next());
				}

				// Remove node from levels.
				this.levels.removeNode(toExpand);

				// Update maxLoopingVal.
				this.maxLoopingVal = curLevelMaxVal;

				// Break out of loop.
				break;
			}

			// Go down a level.
			incrementDepth();
		}

		// Add to expansion count.
		this.expansionCount = this.expansionCount + 1;

		TreeNode toExpand = this.evalutionQueue.remove();
		double[] peval = toExpand.pointOfEvalution();

		// Evaluate functions at point.
		double evaluation = this.evaluate(peval);

		// Add evaluation to node.
		toExpand.setEvalutionValue(evaluation);

		// Add node to the levels.
		this.levels.addNode(toExpand);

		return toExpand;
	}
}

class Levels {
	private ArrayList<Level> levelsList;
	private ArrayList<Double> maxValsUpto;
	private int numLevels = 0;

	public Levels() {
		this.levelsList = new ArrayList<Level>();
		this.maxValsUpto = new ArrayList<Double>();
	}

	public int getNumLevels() {
		return this.numLevels;
	}

	public Level getLevel(int depth) {
		return this.levelsList.get(depth);
	}

	public double getMaxUpto(int depth) {
		return this.maxValsUpto.get(depth).doubleValue();
	}

	public void addNode(TreeNode node) {
		int depth = node.getDepth();
		Level targetLevel = null;

		if (depth < this.levelsList.size()) {
			// If level already exist.
			targetLevel = this.levelsList.get(depth);
		} else {
			// If level does not already exist.
			this.numLevels = this.numLevels + 1;
			targetLevel = new Level();
			this.levelsList.add(targetLevel);
		}
		
		// Add node to the target level
		targetLevel.add(node);

		// Recalculate the max of the level.
		targetLevel.nominate();
	}

	public void removeNode(TreeNode node) {
		int depth = node.getDepth();

		// Remove node from its level.
		Level targetLevel = this.levelsList.get(depth);
		targetLevel.removeNode(node);

		// Recalculate the max of the level.
		targetLevel.nominate();
	}
}

class Level {
	private ArrayList<TreeNode> levelList;
	private double maxVal = Double.NEGATIVE_INFINITY;
	private TreeNode maxNode = null;
	private int depth = -1;

	public Level() {
		this.levelList = new ArrayList<TreeNode>();
	}
	
	public void add(TreeNode node) {
		this.levelList.add(node);
	}

	public void removeNode(TreeNode node) {
		this.levelList.remove(node);
	}

	public void nominate() {
		double max = Double.NEGATIVE_INFINITY;
		TreeNode nominee = null;
		for (int i = 0; i < this.levelList.size(); i++) {
			TreeNode ith = this.levelList.get(i);
			double val = ith.getValue();
			if (val > max) {
				max = val;
				nominee = ith;
			}
		}
		this.maxVal = max;
		this.maxNode = nominee;
	}

	public TreeNode getMaxNode() {
		return this.maxNode;
	}

	public double getMaxVal() {
		return this.maxVal;
	}

}

