/***************************************************************************************************
 *  cosd.swift
 *
 *  This file provides cosine functionality in degrees.
 *
 *  Author: Philip Erickson
 *  Creation Date: 1 May 2016
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 *  in compliance with the License. You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software distributed under the 
 *  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
 *  express or implied. See the License for the specific language governing permissions and 
 *  limitations under the License.
 *
 *  Copyright 2016 Philip Erickson
 **************************************************************************************************/

/// Compute the cosine of x in degrees.
///
/// - Parameters:
///     - x: value to cosine of
/// - Returns: cosine of x
public func cosd(_ x: Double) -> Double 
{
    return cos(x * NiftyNamespace.Constants.pi/180)
}

public func cosd(_ v: Vector<Double>) -> Vector<Double>
{
    let newData = v.data.map({cosd($0)})

    return Vector(newData, name: v.name, showName: v.showName)
}

public func cosd(_ m: Matrix<Double>) -> Matrix<Double>
{
    let newData = m.data.map({cosd($0)})

    return Matrix(m.size, newData, name: m.name, showName: m.showName)
}

public func cosd(_ t: Tensor<Double>) -> Tensor<Double>
{
    let newData = t.data.map({cosd($0)})

    return Tensor(t.size, newData, name: t.name, showName: t.showName)
}