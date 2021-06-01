extension Tensor where Element: Numeric {
  public static func += (lhs: inout Self, rhs: Self) {
    precondition(lhs.size == rhs.size, "sizes must match")

    for i in 0..<lhs.count {
      lhs.data[i] += rhs.data[i]
    }
  }

  public static func + (lhs: Self, rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }
}

extension Tensor where Element: BinaryFloatingPoint {
  /// the two tensors have to be of shape rxn, nxc  
  /// - Result: will have shape rxc
  public func matmul(_ other: Tensor<Element>) -> Tensor<Element> {
    guard dim == 2 else {
      fatalError("both tensors must have 2 dimensions, first one has: \(dim)")
    }
    guard other.dim == 2 else {
      fatalError("both tensors must have 2 dimensions, second one has: \(other.dim)")
    }
    guard shape[1] == other.shape[0] else {
      fatalError("dimensions 1 of first and 0 of second must match in size, are: \(shape[1]), \(other.shape[0])")
    }

    var result = Tensor([shape[0], other.shape[1]], Array<Element>(repeating: 0, count: shape[0] * other.shape[1]))

    for row in 0..<shape[0] {
      for column in 0..<other.shape[1] {
        var dot: Element = 0
        for i in 0..<shape[1] {
          dot += self[row, i] * other[i, column]
        }
        result[row, column] = dot
      }
    }

    return result
  }
}

/// element wise max of both tensors
public func max<E>(_ t1: Tensor<E>, _ t2: Tensor<E>) -> Tensor<E> where E: Comparable {
  precondition(t1.shape == t2.shape, "shapes must match")

  var result = t1

  for i in 0..<t1.count {
    result.data[i] = max(result.data[i], t2.data[i])
  }

  return result
}

/// tensor value if bigger than scalar or scalar if scalar is bigger
public func max<E>(_ t: Tensor<E>, _ s: E) -> Tensor<E> where E: Comparable {
  max(t, Tensor(t.shape, value: s))
}

/// tensor value if bigger than scalar or scalar if scalar is bigger
public func max<E>(_ s: E, _ t: Tensor<E>) -> Tensor<E> where E: Comparable {
  max(t, s)
}
