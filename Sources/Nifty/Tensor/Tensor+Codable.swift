import Foundation

extension Tensor: Codable where Element: Codable {
	public init(from decoder: Decoder) throws {
		let container = try decoder.container(keyedBy: CodingKeys.self)
		let shape = try container.decode([Int].self, forKey: .shape)
		let data = try container.decode([Element].self, forKey: .data)
		self.init(shape, data)
	}

	public func encode(to encoder: Encoder) throws {
		var container = encoder.container(keyedBy: CodingKeys.self)
		try container.encode(shape, forKey: .shape)
		try container.encode(data, forKey: .data)
	}

	public enum CodingKeys: String, CodingKey {
		case shape
		case data 
	}
}