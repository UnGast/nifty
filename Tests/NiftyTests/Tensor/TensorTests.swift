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
}