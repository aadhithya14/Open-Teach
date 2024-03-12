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

public class OVRAudioSourceTest : MonoBehaviour
{
	public float period = 2.0f;
	private float nextActionTime;

	// Start is called before the first frame update
	void Start()
	{
		Material templateMaterial = GetComponent<Renderer>().material;
		Material newMaterial = Instantiate<Material>(templateMaterial);
		newMaterial.color = Color.green;
		GetComponent<Renderer>().material = newMaterial;

		nextActionTime = Time.time + period;
	}

	// Update is called once per frame
	void Update()
	{
		if (Time.time > nextActionTime)
		{
			nextActionTime = Time.time + period;

			Material mat = GetComponent<Renderer>().material;
			if (mat.color == Color.green)
			{
				mat.color = Color.red;
			}
			else
			{
				mat.color = Color.green;
			}

			AudioSource audioSource = GetComponent<AudioSource>();
			if (audioSource == null)
			{
				Debug.LogError("Unable to find AudioSource");
			}
			else
			{
				audioSource.Play();
			}
		}
	}
}
