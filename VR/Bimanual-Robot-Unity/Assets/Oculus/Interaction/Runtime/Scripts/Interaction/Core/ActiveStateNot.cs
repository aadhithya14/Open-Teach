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
    public class ActiveStateNot : MonoBehaviour, IActiveState
    {
        [SerializeField, Interface(typeof(IActiveState))]
        private MonoBehaviour _activeState;

        private IActiveState ActiveState;

        protected virtual void Awake()
        {
            ActiveState = _activeState as IActiveState;;
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(ActiveState);
        }

        public bool Active => !ActiveState.Active;

        #region Inject

        public void InjectAllActiveStateNot(IActiveState activeState)
        {
            InjectActiveState(activeState);
        }

        public void InjectActiveState(IActiveState activeState)
        {
            _activeState = activeState as MonoBehaviour;
            ActiveState = activeState;
        }
        #endregion
    }
}
