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
    public class ActiveStateSelector : MonoBehaviour, ISelector
    {
        [SerializeField, Interface(typeof(IActiveState))]
        private MonoBehaviour _activeState;

        private IActiveState ActiveState;
        private bool _selecting = false;

        public event Action WhenSelected = delegate { };
        public event Action WhenUnselected = delegate { };

        protected virtual void Awake()
        {
            ActiveState = _activeState as IActiveState;
        }

        protected virtual void Start()
        {
            Assert.IsNotNull(ActiveState);
        }

        protected virtual void Update()
        {
            if (_selecting != ActiveState.Active)
            {
                _selecting = ActiveState.Active;
                if (_selecting)
                {
                    WhenSelected();
                }
                else
                {
                    WhenUnselected();
                }
            }
        }

        #region Inject

        public void InjectAllActiveStateSelector(IActiveState activeState)
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
