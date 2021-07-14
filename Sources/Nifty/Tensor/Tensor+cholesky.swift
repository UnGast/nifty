import CLapacke

extension TensorProtocol where Element == Double {
	/// - Returns: the lower triangular matrix obtained from the cholesky decomposition
	public func cholesky() -> Self {
		precondition(shape[0] == shape[1] && shape.count == 2, "cholesky decomposition works only on square matrices")

		var a = data

		let info = LAPACKE_dpotrf(
			LAPACK_ROW_MAJOR,
			76 /* return lower, acii L */,
			Int32(shape[0]),
			&a,
			Int32(shape[1]))
			
		precondition(info >= 0, "Illegal value in LAPACK argument \(-1*info)")
		precondition(info == 0, "The leading minor of order \(info) is not positive definite, and the " +
			"factorization could not be completed")
		
		return Self(shape, a, name: nil, showName: nil)
	}
}