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

namespace Oculus.Interaction.PoseDetection
{
    public class SequenceActiveState : MonoBehaviour, IActiveState
    {
        [SerializeField]
        private Sequence _sequence;

        [SerializeField]
        private bool _activateIfStepsStarted;

        [SerializeField]
        private bool _activateIfStepsComplete = true;

        protected virtual void Start()
        {
            Assert.IsNotNull(_sequence);
        }

        public bool Active
        {
            get
            {
                return (_activateIfStepsStarted && _sequence.CurrentActivationStep > 0 && !_sequence.Active) ||
                       (_activateIfStepsComplete && _sequence.Active);
            }
        }

        #region Inject

        public void InjectAllSequenceActiveState(Sequence sequence,
            bool activateIfStepsStarted, bool activateIfStepsComplete)
        {
            InjectSequence(sequence);
            InjectActivateIfStepsStarted(activateIfStepsStarted);
            InjectActivateIfStepsComplete(activateIfStepsComplete);
        }

        public void InjectSequence(Sequence sequence)
        {
            _sequence = sequence;
        }

        public void InjectActivateIfStepsStarted(bool activateIfStepsStarted)
        {
            _activateIfStepsStarted = activateIfStepsStarted;
        }

        public void InjectActivateIfStepsComplete(bool activateIfStepsComplete)
        {
            _activateIfStepsComplete = activateIfStepsComplete;
        }

        #endregion
    }
}
