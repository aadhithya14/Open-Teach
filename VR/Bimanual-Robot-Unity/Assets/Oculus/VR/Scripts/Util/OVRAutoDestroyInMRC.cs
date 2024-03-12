/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// If there is a game object under the main camera which should not be cloned under Mixed Reality Capture,
// attaching this component would auto destroy that after the MRC camera get cloned
public class OVRAutoDestroyInMRC : MonoBehaviour {

	// Use this for initialization
	void Start () {
		bool underMrcCamera = false;

		Transform p = transform.parent;
		while (p != null)
		{
			if (p.gameObject.name.StartsWith("OculusMRC_"))
			{
				underMrcCamera = true;
				break;
			}
			p = p.parent;
		}

		if (underMrcCamera)
		{
			Destroy(gameObject);
		}
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
