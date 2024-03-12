/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;

public class OVRResources : MonoBehaviour
{
	private static AssetBundle resourceBundle;
	private static List<string> assetNames;

	public static UnityEngine.Object Load(string path)
	{
		if (Debug.isDebugBuild)
		{
			if(resourceBundle == null)
			{
				Debug.Log("[OVRResources] Resource bundle was not loaded successfully");
				return null;
			}

			var result = assetNames.Find(s => s.Contains(path.ToLower()));
			return resourceBundle.LoadAsset(result);
		}
		return Resources.Load(path);
	}
	public static T Load<T>(string path) where T : UnityEngine.Object
	{
		if (Debug.isDebugBuild)
		{
			if (resourceBundle == null)
			{
				Debug.Log("[OVRResources] Resource bundle was not loaded successfully");
				return null;
			}

			var result = assetNames.Find(s => s.Contains(path.ToLower()));
			return resourceBundle.LoadAsset<T>(result);
		}
		return Resources.Load<T>(path);
	}

	public static void SetResourceBundle(AssetBundle bundle)
	{
		resourceBundle = bundle;
		assetNames = new List<string>();
		assetNames.AddRange(resourceBundle.GetAllAssetNames());
	}
}
