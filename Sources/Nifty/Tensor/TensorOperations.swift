extension Tensor {
  /// - Returns: the only element if the tensor only contains one element, otherwise nil
  public var item: Element? {
    if count == 1 {
      return data[0]
    } else {
      return nil
    }
  }

  /// - Returns: same data, but all dimensions with size of 1 removed
  public func squeezed() -> Self {
    Self(shape.filter { $0 != 1 }, data)
  }

  /// - Returns: Same data. One dimension added at specified index.
  /// Dimension at that index moves to the right.
  public func unsqueezed(dim: Int) -> Self {
    var newShape = shape
    newShape.insert(1, at: dim)
    return Self(newShape, data)
  }

  /// works only for 2 dimensional tensors (matrices)
  /// - Returns: swapped columns and rows
  public func transposed() -> Self {
    precondition(dim == 2, "transposed() only works for matrices (2 dimensional tensors)")

    var swappedData = [Element]()
    swappedData.reserveCapacity(numel)

    let nRows = shape[0]
    let nColumns = shape[1]

    for x in 0..<nColumns {
      for y in 0..<nRows {
        swappedData.append(data[y * nColumns + x])
      }
    }

    return Self([nColumns, nRows], swappedData)
  }

  public func dataConverted<T>(_ convert: (Element) -> T) -> Tensor<T> {
    Tensor<T>(shape, data.map(convert))
  }
}

extension Tensor: Equatable where Element: Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.data == rhs.data
  }
}

extension Tensor: Hashable where Element: Hashable {
  public func hash(into hasher: inout Hasher) {
    hasher.combine(data)
  }
}

extension Tensor where Element: Numeric {
  public static func identityMatrix(_ n: Int) -> Self {
    var elements = [Element]()
    elements.reserveCapacity(n * n)
    for x in 0..<n {
      for y in 0..<n {
        elements.append(x == y ? 1 : 0)
      }
    }
    return Tensor([n, n], elements)
  }

  /// - Returns: true if shape equals and element wise absolute difference <= within, false otherwise
  public func isEqual(to other: Self, within threshold: Element) -> Bool where Element: Comparable {
    if shape != other.shape {
      return false
    }

    let diff = (self - other).abs()
    for i in 0..<numel {
      if diff.data[i] > threshold {
        return false
      }
    }
    
    return true
  }

  @inlinable public static func += (lhs: inout Self, rhs: Self) {
    precondition(lhs.size == rhs.size, "sizes must match")

    for i in 0..<lhs.count {
      lhs.data[i] += rhs.data[i]
    }
  }

  @inlinable public static func + (lhs: Self, rhs: Self) -> Self {
    var result = lhs
    result += rhs
    return result
  }

  @inlinable public static func -= (lhs: inout Self, rhs: Self) {
    precondition(lhs.size == rhs.size, "sizes must match")

    for i in 0..<lhs.count {
      lhs.data[i] -= rhs.data[i]
    }
  }

  @inlinable public static func - (lhs: Self, rhs: Self) -> Self {
    var result = lhs
    result -= rhs
    return result
  }

  /// element wise multiplication
  @inlinable public static func *= (lhs: inout Self, rhs: Self) {
    precondition(lhs.size == rhs.size, "sizes must match")

    for i in 0..<lhs.count {
      lhs.data[i] *= rhs.data[i]
    }
  }

  /// element wise multiplication
  @inlinable public static func * (lhs: Self, rhs: Self) -> Self {
    var result = lhs
    result *= rhs
    return result
  }

  @inlinable public static func *= (lhs: inout Self, rhs: Element) {
    for i in 0..<lhs.count {
      lhs.data[i] *= rhs
    }
  }

  @inlinable public static func * (lhs: Self, rhs: Element) -> Self {
    var result = lhs
    result *= rhs
    return result
  }

  @inlinable public static func * (lhs: Element, rhs: Self) -> Self {
    rhs * lhs
  }

  /// - Returns: sum of all elements
  public func sum() -> Element {
    data.reduce(into: Element.zero) { $0 += $1 }
  }

  /// - Returns: element wise abs
  public func abs() -> Self where Element: SignedNumeric, Element: Comparable {
    var result = self
    for i in 0..<count {
      result.data[i] = Swift.abs(result.data[i])
    }
    return result
  }

  /// - Returns: element wise abs, same as when not applied, because using on non-signed numeric value
  public func abs() -> Self where Element: Comparable {
    return self
  }
}

extension Tensor where Element: FloatingPoint {
  public static func /= (lhs: inout Self, rhs: Self) {
    precondition(lhs.size == rhs.size, "sizes must match")

    for i in 0..<lhs.count {
      lhs.data[i] /= rhs.data[i]
    }
  }

  public static func / (lhs: Self, rhs: Self) -> Self {
    var result = lhs
    result /= rhs
    return result
  }

  public static func /= (lhs: inout Self, rhs: Element) {
    for i in 0..<lhs.count {
      lhs.data[i] /= rhs
    }
  }

  public static func / (lhs: Self, rhs: Element) -> Self {
    var result = lhs
    result /= rhs
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

  public static func random(_ shape: [Int], in range: Range<Element>) -> Tensor<Element> where Element.RawSignificand: FixedWidthInteger {
    Tensor(shape, (0..<calcNumel(shape: shape)).map { _ in Element.random(in: range) })
  }
}

extension Tensor where Element: Comparable {
  /// - Returns: maximum element in whole tensor
  public func max() -> Element {
    data.reduce(into: Optional<Element>.none) {
      if $0 == nil || $0! < $1 {
        $0 = $1
      }
    }!
  }
}

/// - Returns: element wise max of both tensors
public func max<E>(_ t1: Tensor<E>, _ t2: Tensor<E>) -> Tensor<E> where E: Comparable {
  precondition(t1.shape == t2.shape, "shapes must match")

  var result = t1

  for i in 0..<t1.count {
    result.data[i] = max(result.data[i], t2.data[i])
  }

  return result
}

/// - Returns: tensor value if bigger than scalar or scalar if scalar is bigger
public func max<E>(_ t: Tensor<E>, _ s: E) -> Tensor<E> where E: Comparable {
  max(t, Tensor(t.shape, value: s))
}

/// - Returns: tensor value if bigger than scalar or scalar if scalar is bigger
public func max<E>(_ s: E, _ t: Tensor<E>) -> Tensor<E> where E: Comparable {
  max(t, s)
}