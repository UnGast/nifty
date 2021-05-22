// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "Nifty",
    products: [
        .library(name: "Nifty", targets: ["Nifty"])
    ],
    dependencies: [
        .package(path: "../Nifty-libs"),
    ],
    targets: [
        .target(name: "Nifty", dependencies: [.product(name: "CLapacke", package: "Nifty-libs"), .product(name: "CBlas", package: "Nifty-libs")])
    ]
)