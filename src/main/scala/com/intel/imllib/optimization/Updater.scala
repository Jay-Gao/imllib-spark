package com.intel.imllib.optimization

import scala.math._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import com.intel.imllib.util.vectorUtils._
trait Updater extends Serializable {
	val epsilon = 1e-8
}


/**
	* :: DeveloperApi ::
	* A simple updater for gradient descent with L2 regularization.
	*/
@DeveloperApi
class SimpleUpdater(
										 private var learningRate: Double,
										 private var l2RegParam: Double
									 ) extends Updater {
	require(this.l2RegParam >= 0, s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
	def this(lr: Double) = this(lr, 0.0)
	def this() = this(0.01)

	def setLearningRate(lr: Double) = {
		this.learningRate = lr
		this
	}

	def setRegParam(regParam: Double) = {
		require(this.l2RegParam >= 0,
			s"Regularization parameter must be nonnegative but got ${this.l2RegParam}")
		this.l2RegParam = regParam
		this
	}

	def compute(
							 weightsOld: Vector,
							 gradients: Vector
						 ): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val weights = toBreeze(weightsOld) - this.learningRate * actualGradients
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}

@DeveloperApi
class MomentumUpdater(
											 private var learningRate: Double,
											 private var gamma: Double,
											 private var l2RegParam: Double
										 ) extends Updater {
	require(l2RegParam >= 0,
		s"Regularization parameter must be nonnegative but got $l2RegParam")
	private var mementum: Vector = null
	def this(lr: Double, gamma: Double) = this(lr, gamma, 0.0)
	def this(lr: Double) = this(lr, 0.9, 0.0)
	def this() = this(0.01)

	def setLearningRate(lr: Double) = {
		this.learningRate = lr
		this
	}

	def setGamma(gamma: Double) = {
		this.gamma = gamma
		this
	}

	def setRegParam(regParam: Double) = {
		require(regParam >= 0,
			s"Regularization parameter must be nonnegative but got $regParam")
		this.l2RegParam = regParam
		this
	}

	def initializeMomentum(N: Int) = {
		this.mementum = Vectors.zeros(N)
		this
	}

	def compute(
							 weightsOld: Vector,
							 gradients: Vector
						 ): (Vector, Double) = {
		val actualGradients = toBreeze(gradients) + this.l2RegParam * toBreeze(weightsOld)
		val accum = toBreeze(this.mementum) * this.gamma + this.learningRate * actualGradients
		val weights = toBreeze(weightsOld) - accum
		this.mementum = fromBreeze(accum)
		val regLoss = if (this.l2RegParam == 0) 0 else 0.5 * this.l2RegParam * (toBreeze(weightsOld) :* toBreeze(weightsOld)).sum
		(fromBreeze(weights), regLoss)
	}
}
