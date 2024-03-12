/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

#if USING_XR_SDK_OPENXR

using System.IO;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEngine.XR.OpenXR;
using UnityEditor.XR.OpenXR.Features;

namespace Oculus.XR
{
	/// <summary>
	/// Automatically enables the OculusXRFeature feature
	/// </summary>
	[InitializeOnLoad]
	public class OculusXRFeatureEnabler : MonoBehaviour
	{
		static OculusXRFeatureEnabler()
		{
			EditorApplication.update += EnableOculusXRFeature;
		}

		private static void EnableOculusXRFeature()
		{
			EditorApplication.update -= EnableOculusXRFeature;

			bool unityRunningInBatchmode = false;

			if (System.Environment.CommandLine.Contains("-batchmode"))
			{
				unityRunningInBatchmode = true;
			}

			bool needEnable = false;

			var featureStandalone = FeatureHelpers.GetFeatureWithIdForBuildTarget(BuildTargetGroup.Standalone, OculusXRFeature.featureId);
			var featureAndroid = FeatureHelpers.GetFeatureWithIdForBuildTarget(BuildTargetGroup.Android, OculusXRFeature.featureId);

			if (featureStandalone != null && !featureStandalone.enabled)
				needEnable = true;

			if (featureAndroid != null && !featureAndroid.enabled)
				needEnable = true;

			if (needEnable && !unityRunningInBatchmode)
			{
				bool result = EditorUtility.DisplayDialog("Enable OculusXR Feature", "OculusXR Feature must be enabled in OpenXR Feature Groups to support Oculus Utilities. Do you want to enable it now?", "Enable", "Cancel");
				if (!result)
				{
					needEnable = false;
					EditorUtility.DisplayDialog("OculusXR Feature not enabled", "You can enable OculusXR Feature in XR Plugin-in Management / OpenXR for using Oculus Utilities functionalities. Please enable it in both Standalone and Android settings.", "Ok");
				}
			}

			if (needEnable)
			{
				if (featureStandalone != null && !featureStandalone.enabled)
				{
					Debug.Log("OculusXRFeature enabled on Standalone");
					featureStandalone.enabled = true;
				}
				if (featureAndroid != null && !featureAndroid.enabled)
				{
					Debug.Log("OculusXRFeature enabled on Android");
					featureAndroid.enabled = true;
				}
			}
		}
	}
}

#endif
