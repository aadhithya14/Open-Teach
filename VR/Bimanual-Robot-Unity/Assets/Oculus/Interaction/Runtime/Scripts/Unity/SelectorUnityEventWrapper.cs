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
using UnityEngine.Assertions;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    public class SelectorUnityEventWrapper : MonoBehaviour
    {
        [SerializeField, Interface(typeof(ISelector))]
        private MonoBehaviour _selector;
        private ISelector Selector;

        [SerializeField]
        private UnityEvent _whenSelected;

        [SerializeField]
        private UnityEvent _whenUnselected;

        public UnityEvent WhenSelected => _whenSelected;
        public UnityEvent WhenUnselected => _whenUnselected;

        protected bool _started = false;

        protected virtual void Awake()
        {
            Selector = _selector as ISelector;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(Selector);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                Selector.WhenSelected += HandleSelected;
                Selector.WhenUnselected += HandleUnselected;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                Selector.WhenSelected -= HandleSelected;
                Selector.WhenUnselected -= HandleUnselected;
            }
        }

        private void HandleSelected()
        {
            _whenSelected.Invoke();
        }

        private void HandleUnselected()
        {
            _whenUnselected.Invoke();
        }

        #region Inject

        public void InjectAllSelectorUnityEventWrapper(ISelector selector)
        {
            InjectSelector(selector);
        }

        public void InjectSelector(ISelector selector)
        {
            _selector = selector as MonoBehaviour;
            Selector = selector;
        }

        #endregion
    }
}
