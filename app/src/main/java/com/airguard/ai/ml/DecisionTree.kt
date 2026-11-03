package com.airguard.ai.ml

/**
 * A single decision tree node, used by the Random Forest.
 * This is a pure Kotlin implementation — no native library required.
 * Trees are encoded as immutable data structures built at runtime from
 * the training results baked in below.
 */
sealed class TreeNode {
    data class Split(
        val featureIndex: Int,
        val threshold: Double,
        val left: TreeNode,   // feature[featureIndex] <= threshold
        val right: TreeNode   // feature[featureIndex] > threshold
    ) : TreeNode()

    data class Leaf(val classId: Int) : TreeNode()
}

/**
 * Evaluates a single decision tree.
 */
object DecisionTreeEvaluator {
    fun predict(node: TreeNode, features: DoubleArray): Int {
        var current = node
        while (current is TreeNode.Split) {
            current = if (features[current.featureIndex] <= current.threshold) {
                current.left
            } else {
                current.right
            }
        }
        return (current as TreeNode.Leaf).classId
    }
}
