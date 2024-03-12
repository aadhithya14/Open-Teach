/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;

public class OVRPointerVisualizer : MonoBehaviour
{
	[Tooltip("Object which points with Z axis. E.g. CentreEyeAnchor from OVRCameraRig")]
	public Transform rayTransform;
	[Header("Visual Elements")]
	[Tooltip("Line Renderer used to draw selection ray.")]
	public LineRenderer linePointer = null;
	[Tooltip("Visually, how far out should the ray be drawn.")]
	public float rayDrawDistance = 2.5f;

	void Update()
	{
		linePointer.enabled = (OVRInput.GetActiveController() == OVRInput.Controller.Touch);
		Ray ray = new Ray(rayTransform.position, rayTransform.forward);
		linePointer.SetPosition(0, ray.origin);
		linePointer.SetPosition(1, ray.origin + ray.direction * rayDrawDistance);
	}
}
