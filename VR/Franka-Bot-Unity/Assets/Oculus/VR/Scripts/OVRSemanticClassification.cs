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
using UnityEngine;

/// <summary>
/// Represents the semantic classification of a <see cref="OVRSceneAnchor"/>.
/// </summary>
/// <remarks>
/// Scene anchors can have one or more string labels associated with them that describes what the anchor represents,
/// such as COUCH, or DESK. See <see cref="OVRSceneManager.Classification"/> for a list of possible labels.
/// </remarks>
[DisallowMultipleComponent]
[RequireComponent(typeof(OVRSceneAnchor))]
public class OVRSemanticClassification : MonoBehaviour, IOVRSceneComponent
{
	private readonly List<string> _labels = new List<string>();

	/// <summary>
	/// A list of labels associated with an <see cref="OVRSceneAnchor"/>.
	/// </summary>
	/// <exception cref="ArgumentNullException"></exception>
	public IReadOnlyList<string> Labels => _labels;

	/// <summary>
	/// Searches <see cref="Labels"/> for the given <paramref name="label"/>.
	/// </summary>
	/// <remarks>
	/// This method performs a linear search over the <see cref="Labels"/>.
	/// </remarks>
	/// <param name="label">The label to find</param>
	/// <returns>Returns true if <paramref name="label"/> exists in <see cref="Labels"/>.</returns>
	public bool Contains(string label)
	{
		foreach (var item in _labels)
		{
			if (item == label)
			{
				return true;
			}
		}

		return false;
	}

	private void Awake()
	{
		if (GetComponent<OVRSceneAnchor>().Space.Valid)
		{
			((IOVRSceneComponent)this).Initialize();
		}
	}

	void IOVRSceneComponent.Initialize()
	{
		if (OVRPlugin.GetSpaceSemanticLabels(GetComponent<OVRSceneAnchor>().Space, out var labels))
		{
			_labels.Clear();
			_labels.AddRange(labels.Split(','));
		}
		else
		{
			OVRSceneManager.Development.LogWarning(nameof(OVRSemanticClassification),
				$"[{GetComponent<OVRSceneAnchor>().Uuid}] {nameof(OVRSceneAnchor)} has no semantic labels.");
		}
	}
}
