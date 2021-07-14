import XCTest
@testable import Nifty

class TensorCholeskyTests: XCTestCase {
	func testCholesky() {
		let tensor = Tensor<Double>([3, 3], [
			2, -1, 0,
			-1, 3, -1,
			0, -1, 4
		])
		XCTAssertTrue(tensor.cholesky().isEqual(to: Tensor<Double>([3, 3], [
			1.4142, 0, 0,
			-0.7071, 1.5811, 0,
			0, -0.6325, 1.8974
		]), within: 0.01))
	}
}