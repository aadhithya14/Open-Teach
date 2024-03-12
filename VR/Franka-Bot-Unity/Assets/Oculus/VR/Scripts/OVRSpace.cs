/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// Represents a "space" in the Oculus Runtime.
/// </summary>
public readonly struct OVRSpace : IEquatable<OVRSpace>
{
	/// <summary>
	/// A runtime handle associated with this <see cref="OVRSpace"/>. This could change between subsequent sessions
	/// or apps.
	/// </summary>
	public ulong Handle { get; }

	/// <summary>
	/// Indicates whether this <see cref="OVRSpace"/> represents a valid space (vs a default constructed
	/// <see cref="OVRSpace"/>).
	/// </summary>
	public bool Valid => Handle != 0;

	/// <summary>
	/// Constructs an <see cref="OVRSpace"/> object from an existing runtime handle and UUID.
	/// </summary>
	/// <remarks>
	/// This constructor does not create a new space. An <see cref="OVRSpace"/> is a wrapper for low-level functionality
	/// in the Oculus Runtime. To create a new spatial anchor, use <see cref="OVRSpatialAnchor"/>.
	/// </remarks>
	/// <param name="handle">The runtime handle associated with the space.</param>
	public OVRSpace(ulong handle) => Handle = handle;

	/// <summary>
	/// Generates a string representation of this <see cref="OVRSpace"/> of the form
	/// "0xYYYYYYYY"
	/// where "Y" are the hexadecimal digits of the <see cref="Handle"/>.
	/// </summary>
	/// <returns>Returns a string representation of this <see cref="OVRSpace"/>.</returns>
	public override string ToString() => $"0x{Handle:x8}";

	public bool Equals(OVRSpace other) => Handle == other.Handle;

	public override bool Equals(object obj) => obj is OVRSpace other && Equals(other);

	public override int GetHashCode() => Handle.GetHashCode();

	public static bool operator== (OVRSpace lhs, OVRSpace rhs) => lhs.Handle == rhs.Handle;

	public static bool operator!= (OVRSpace lhs, OVRSpace rhs) => lhs.Handle != rhs.Handle;

	public static implicit operator OVRSpace(ulong handle) => new OVRSpace(handle);

	public static implicit operator ulong(OVRSpace space) => space.Handle;
}
