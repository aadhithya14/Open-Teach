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

namespace Oculus.Interaction.Samples
{
    public class EnableTargetOnStart : MonoBehaviour
    {
        public MonoBehaviour[] _components;
        public GameObject[] _gameObjects;

        void Start()
        {
            if (_components != null)
            {
                foreach (MonoBehaviour target in _components)
                {
                    target.enabled = true;
                }
            }

            if (_gameObjects != null)
            {
                foreach (GameObject target in _gameObjects)
                {
                    target.SetActive(true);
                }
            }
        }
    }
}
