import XCTest
@testable import Nifty

class TensorTests: XCTestCase {
	func testIsEqualWithinNonSignedNumeric() {
		let tensor1 = Tensor<UInt>([4], [1, 2, 3, 4])
		let tensor2 = Tensor<UInt>([4], [1, 2, 3, 4])
		XCTAssertTrue(tensor1.isEqual(to: tensor2, within: 1))
	}

	func testIsEqualWithinSignedNumeric() {
		let tensor1 = Tensor<Double>([4], [1, -2, 3, 4])
		let tensor2 = Tensor<Double>([4], [1.01, -1.99, 3.02, 3.98])
		XCTAssertTrue(tensor1.isEqual(to: tensor2, within: 0.1))
		let tensor3 = Tensor<Double>([4], [1.01, -1.99, 3.5, 3.98])
		XCTAssertFalse(tensor1.isEqual(to: tensor3, within: 0.01))
	}

	func testTransposed() {
		XCTAssertEqual(Tensor([2, 4], [
			1, 2, 3, 4,
			5, 6, 7, 8
		]).transposed(), Tensor([4, 2], [
			1, 5,
			2, 6,
			3, 7,
			4, 8
		]))
	}

	func testAllSqueezed() {
		let tensor = Tensor<Double>([2, 1, 2, 1], [
			1, 2,
			3, 4,
		])
		let squeezed = tensor.squeezed()
		XCTAssertEqual(tensor.data, squeezed.data)
		XCTAssertEqual(squeezed.shape, [2, 2])
	}

	func testUnsqueezed() {
		let tensor = Tensor([2, 3, 1], [
			1, 2, 3,
			4, 5, 6
		])
		let unsqueezed = tensor.unsqueezed(dim: 1)
		XCTAssertEqual(unsqueezed.shape, [2, 1, 3, 1])
		XCTAssertEqual(tensor.data, unsqueezed.data)
	}

	func testTensorTensorAddition() {
		let tensor1 = Tensor([3, 1], [1, 2, 3])
		let tensor2 = Tensor([3, 1], [2, 3, 4])
		let targetResult = Tensor([3, 1], [3, 5, 7])
		var testResult1 = tensor1
		testResult1 += tensor2
		let testResult2 = tensor1 + tensor2
		XCTAssertEqual(testResult1, targetResult)
		XCTAssertEqual(testResult2, targetResult)
	}

	/*func testTensorScalarAddition() {
		let tensor = Tensor([3, 2], [1, 2, 3, 4, 5, 6])
		let scalar = 2
		let targetResult = Tensor([3, 2], [3, 4, 5, 6, 7, 8])
		var testResult1 = tensor
		testResult1 += scalar
		let testResult2 = tensor + scalar
		let testResult3 = scalar + tensor
		XCTAssertEqual(targetResult, testResult1)
		XCTAssertEqual(targetResult, testResult2)
		XCTAssertEqual(targetResult, testResult3)
	}*/

	func testTensorTensorSubtraction() {
		let tensor1 = Tensor([3, 1], [1, 2, 3])
		let tensor2 = Tensor([3, 1], [2, 3, 4])
		let targetResult = Tensor([3, 1], [-1, -1, -1])
		var testResult1 = tensor1
		testResult1 -= tensor2
		let testResult2 = tensor1 - tensor2
		XCTAssertEqual(testResult1, targetResult)
		XCTAssertEqual(testResult2, targetResult)
	}

	func testTensorTensorMultiplication() {
		let tensor1 = Tensor([3, 1], [1, 2, 3])
		let tensor2 = Tensor([3, 1], [2, 3, 4])
		let targetResult = Tensor([3, 1], [2, 6, 12])
		var testResult1 = tensor1
		testResult1 *= tensor2
		let testResult2 = tensor1 * tensor2
		XCTAssertEqual(testResult1, targetResult)
		XCTAssertEqual(testResult2, targetResult)
	}

	func testTensorScalarMultiplication() {
		let tensor = Tensor([3, 2], [1, 2, 3, 4, 5, 6])
		let scalar = 2
		let targetResult = Tensor([3, 2], [2, 4, 6, 8, 10, 12])
		var testResult1 = tensor
		testResult1 *= scalar
		let testResult2 = tensor * scalar
		let testResult3 = scalar * tensor
		XCTAssertEqual(targetResult, testResult1)
		XCTAssertEqual(targetResult, testResult2)
		XCTAssertEqual(targetResult, testResult3)
	}

	func testTensorTensorDivision() {
		let tensor1 = Tensor<Double>([3, 1], [1, 2, 3])
		let tensor2 = Tensor<Double>([3, 1], [2, 3, 4])
		let targetResult = Tensor<Double>([3, 1], [0.5, 2.0/3, 3.0/4])
		var testResult1 = tensor1
		testResult1 /= tensor2
		let testResult2 = tensor1 / tensor2
		XCTAssertTrue(testResult1.isEqual(to: targetResult, within: 0.01))
		XCTAssertTrue(testResult2.isEqual(to: targetResult, within: 0.01))
	}
}