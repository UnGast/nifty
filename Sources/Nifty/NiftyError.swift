public struct NiftyError: Error {
  public let message: String

  public init(_ message: String) {
    self.message = message
  }
}