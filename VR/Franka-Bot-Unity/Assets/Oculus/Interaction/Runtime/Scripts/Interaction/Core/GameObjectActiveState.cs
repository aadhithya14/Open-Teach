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
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    public class GameObjectActiveState : MonoBehaviour, IActiveState
    {
        [SerializeField]
        private GameObject _sourceGameObject;

        [SerializeField]
        private bool _sourceActiveSelf;

        public bool SourceActiveSelf
        {
            get
            {
                return _sourceActiveSelf;
            }
            set
            {
                _sourceActiveSelf = value;
            }
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(_sourceGameObject);
        }

        public bool Active => _sourceActiveSelf
            ? _sourceGameObject.activeSelf
            : _sourceGameObject.activeInHierarchy;

        #region Inject

        public void InjectAllGameObjectActiveState(GameObject sourceGameObject)
        {
            InjectSourceGameObject(sourceGameObject);
        }

        public void InjectSourceGameObject(GameObject sourceGameObject)
        {
            _sourceGameObject = sourceGameObject;
        }

        #endregion
    }
}
