func calcNumel(shape: [Int]) -> Int {
  shape.reduce(1) { $0 * $1 }
}