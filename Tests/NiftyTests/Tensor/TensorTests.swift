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
}